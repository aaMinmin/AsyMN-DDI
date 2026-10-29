import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


@tf.keras.utils.register_keras_serializable(package="DRAM_DDI")
class AsyMNFusion(keras.layers.Layer):
    """DRAM-DDI's DAM module: MI-based cross-modal redundancy, local consistency,
    asymmetric redundancy allocation, learnable ATT/RED blending, and feature-wise gating.

    Inputs: [motif_a, motif_b, kg_a, kg_b, kg_pair, fp_a, fp_b] and optionally
    [motif_mask_a, motif_mask_b]. Statistics must be computed from TRAIN data only.
    """
    def __init__(self, feature_dim, stats_a_path, stats_b_path,
                 mi_temperature=1.0, local_temperature=1.0,
                 responsibility_temperature=0.2, redundancy_weight=0.005,
                 detach_att_for_red=True, ema_decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.stats_a_path = stats_a_path
        self.stats_b_path = stats_b_path
        self.mi_temperature = float(mi_temperature)
        self.local_temperature = float(local_temperature)
        self.responsibility_temperature = float(responsibility_temperature)
        self.redundancy_weight = float(redundancy_weight)
        self.detach_att_for_red = bool(detach_att_for_red)
        self.ema_decay = float(ema_decay)

        if self.local_temperature <= 0 or self.mi_temperature <= 0 or self.responsibility_temperature <= 0:
            raise ValueError("temperature values must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")

        self._cross_stats = {}
        self._ema_mean = {}
        self._ema_var = {}
        for role, path in (("a", stats_a_path), ("b", stats_b_path)):
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Missing {role}-side train statistics: {path}")
            with np.load(path, allow_pickle=False) as data:
                required = {"mean", "std", "edge_m", "edge_n", "edge_d", "edge_k", "edge_rho", "edge_mi"}
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"Statistics file {path} missing fields: {sorted(missing)}")
                stats_np = {k: np.asarray(data[k]) for k in required}

            self._cross_stats[role] = {k: tf.constant(v) for k, v in stats_np.items()}

            # One scalar EMA mean/variance per matched edge.
            # Formula (7) supplies the initial variance; the mean starts at zero.
            rho_init = np.asarray(stats_np["edge_rho"], dtype=np.float32)
            theoretical_var = (2.0 * (1.0 - np.abs(rho_init))).astype(np.float32)
            self._ema_mean[role] = self.add_weight(
                name=f"residual_ema_mean_{role}",
                shape=theoretical_var.shape,
                dtype=tf.float32,
                initializer="zeros",
                trainable=False,
            )
            self._ema_var[role] = self.add_weight(
                name=f"residual_ema_var_{role}",
                shape=theoretical_var.shape,
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(theoretical_var),
                trainable=False,
            )

        self.motif_proj_a = layers.Dense(feature_dim, kernel_regularizer=l2(1e-7), name="motif_proj_a")
        self.motif_proj_b = layers.Dense(feature_dim, kernel_regularizer=l2(1e-7), name="motif_proj_b")
        self.kg_proj = layers.Dense(feature_dim, kernel_regularizer=l2(1e-7), name="kg_proj")
        self.fp_proj_a = layers.Dense(feature_dim, kernel_regularizer=l2(1e-7), name="fp_proj_a")
        self.fp_proj_b = layers.Dense(feature_dim, kernel_regularizer=l2(1e-7), name="fp_proj_b")
        
        self.modal_attention = keras.Sequential([
            layers.Dense(feature_dim, activation="relu", kernel_regularizer=l2(1e-7)),
            layers.Dropout(0.2),
            layers.Dense(3 * feature_dim, activation="sigmoid", kernel_regularizer=l2(1e-7)),
        ], name="feature_importance_attention")
        
        # Learnable global blend of task-driven attention and redundancy weights.
        self.alpha_modal_logit = self.add_weight(
            name="att_red_blend_logit", shape=(),
            initializer=tf.keras.initializers.Constant(0.4054651), trainable=True)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"feature_dim": self.feature_dim, 
                    "stats_a_path": self.stats_a_path,
                    "stats_b_path": self.stats_b_path, 
                    "mi_temperature": self.mi_temperature,
                    "local_temperature": self.local_temperature,
                    "responsibility_temperature": self.responsibility_temperature,
                    "redundancy_weight": self.redundancy_weight,
                    "detach_att_for_red": self.detach_att_for_red,
                    "ema_decay": self.ema_decay})
        return cfg

    @staticmethod
    def _pool(x, mask=None):
        x = tf.convert_to_tensor(x)
        if x.shape.rank == 3:
            if mask is None:
                return tf.reduce_mean(x, axis=1)
            valid = tf.cast(mask[..., None], x.dtype)
            return tf.math.divide_no_nan(tf.reduce_sum(x * valid, axis=1), tf.reduce_sum(valid, axis=1))
        return x

    def _update_residual_ema(self, residual, role, training, local_consistency):
        """Update per-edge scalar residual EMA after computing this batch's kernel."""
        if training is None:
            return local_consistency

        def update():
            batch_mean = tf.reduce_mean(residual, axis=0)
            batch_var = tf.reduce_mean(
                tf.square(residual - batch_mean[None, :]), axis=0
            )

            mean_state = self._ema_mean[role]
            var_state = self._ema_var[role]
            decay = tf.cast(self.ema_decay, residual.dtype)

            next_mean = decay * tf.cast(mean_state, residual.dtype) + (
                1.0 - decay
            ) * batch_mean
            next_var = decay * tf.cast(var_state, residual.dtype) + (
                1.0 - decay
            ) * batch_var

            # Update only after this batch's kernel has read the previous EMA state.
            with tf.control_dependencies([local_consistency]):
                assign_mean = mean_state.assign(tf.cast(next_mean, mean_state.dtype))
                assign_var = var_state.assign(tf.cast(next_var, var_state.dtype))
            with tf.control_dependencies([assign_mean, assign_var]):
                return tf.identity(local_consistency)

        if isinstance(training, (bool, np.bool_)):
            return update() if bool(training) else local_consistency

        return tf.cond(
            tf.cast(training, tf.bool),
            update,
            lambda: tf.identity(local_consistency),
        )

    def _redundancy_weights(self, modal_features, att_weights, role, training=None):
        stats = self._cross_stats[role]
        x = tf.stack(modal_features, axis=1)  # [batch, 3 modalities, feature_dim]
        mean = tf.cast(stats["mean"], x.dtype)
        std = tf.maximum(tf.cast(stats["std"], x.dtype), 1e-4)
        z = (x - mean[None, :, :]) / std[None, :, :]
        
        em, en = tf.cast(stats["edge_m"], tf.int32), tf.cast(stats["edge_n"], tf.int32)
        ed, ek = tf.cast(stats["edge_d"], tf.int32), tf.cast(stats["edge_k"], tf.int32)
        d = tf.shape(x)[-1]
        src_idx, dst_idx = em * d + ed, en * d + ek
        flat = tf.reshape(z, [tf.shape(z)[0], -1])
        src, dst = tf.gather(flat, src_idx, axis=1), tf.gather(flat, dst_idx, axis=1)
        rho = tf.cast(stats["edge_rho"], x.dtype)[None, :]
        mi = tf.maximum(tf.cast(stats["edge_mi"], x.dtype), 0.0)[None, :]
        
        # Keep the signed residual and Gaussian kernel; use per-edge EMA
        # mean/variance to standardize the residual dynamically.
        residual = src - tf.sign(rho) * dst
        residual_mean = tf.identity(tf.cast(self._ema_mean[role], x.dtype))
        residual_var = tf.identity(tf.cast(self._ema_var[role], x.dtype))
        squared_distance = tf.square(residual - residual_mean[None, :]) / (
            residual_var[None, :] + 1e-6
        )
        local_consistency = tf.exp(
            -0.5 * squared_distance / (self.local_temperature ** 2)
        )
        # This batch uses the previous EMA snapshot; update afterward.
        local_consistency = self._update_residual_ema(
            residual, role, training, local_consistency
        )
        
        edge_strength = (1.0 - tf.exp(-mi / self.mi_temperature)) * local_consistency
        att = tf.reshape(att_weights, [tf.shape(x)[0], -1])
        if self.detach_att_for_red:
            att = tf.stop_gradient(att)
        att_src, att_dst = tf.gather(att, src_idx, axis=1), tf.gather(att, dst_idx, axis=1)
        
        # Asymmetric allocation: the more important endpoint receives less suppression.
        pi_src = tf.sigmoid((att_dst - att_src) / self.responsibility_temperature)
        score_src, score_dst = edge_strength * pi_src, edge_strength * (1.0 - pi_src)
        
        src_hot = tf.one_hot(src_idx, depth=3 * self.feature_dim, dtype=x.dtype)
        dst_hot = tf.one_hot(dst_idx, depth=3 * self.feature_dim, dtype=x.dtype)
        summed = tf.matmul(score_src, src_hot) + tf.matmul(score_dst, dst_hot)
        counts = tf.reduce_sum(src_hot + dst_hot, axis=0)[None, :]
        redundancy = tf.reshape(tf.math.divide_no_nan(summed, counts), [-1, 3, self.feature_dim])
        
        return tf.exp(-redundancy), redundancy

    def call(self, inputs, training=None):
        if len(inputs) == 9:
            motif_a, motif_b, kg_a, kg_b, kg_pair, fp_a, fp_b, mask_a, mask_b = inputs
        elif len(inputs) == 7:
            motif_a, motif_b, kg_a, kg_b, kg_pair, fp_a, fp_b = inputs
            mask_a = mask_b = None
        else:
            raise ValueError("AsyMNFusion expects 7 inputs, optionally followed by 2 motif masks")
            
        motif_a, motif_b = self._pool(motif_a, mask_a), self._pool(motif_b, mask_b)
        kg_a, kg_b, kg_pair = self._pool(kg_a), self._pool(kg_b), self._pool(kg_pair)
        fp_a, fp_b = self._pool(fp_a), self._pool(fp_b)
        
        mod_a = [self.motif_proj_a(motif_a), self.kg_proj(kg_a), self.fp_proj_a(fp_a)]
        mod_b = [self.motif_proj_b(motif_b), self.kg_proj(kg_b), self.fp_proj_b(fp_b)]
        
        att_a = tf.reshape(self.modal_attention(tf.concat(mod_a, axis=-1), training=training), [-1, 3, self.feature_dim])
        att_b = tf.reshape(self.modal_attention(tf.concat(mod_b, axis=-1), training=training), [-1, 3, self.feature_dim])
        
        red_a, raw_red_a = self._redundancy_weights(mod_a, att_a, "a", training=training)
        red_b, raw_red_b = self._redundancy_weights(mod_b, att_b, "b", training=training)
        
        alpha = tf.sigmoid(self.alpha_modal_logit)
        weights_a = alpha * att_a + (1.0 - alpha) * red_a
        weights_b = alpha * att_b + (1.0 - alpha) * red_b
        
        z_a = tf.reduce_mean(tf.stack(mod_a, axis=1) * weights_a, axis=1)
        z_b = tf.reduce_mean(tf.stack(mod_b, axis=1) * weights_b, axis=1)
        
        pair_h = tf.concat([z_a, z_b, tf.abs(z_a-z_b), z_a*z_b, kg_pair], axis=-1)
        
        # [修复核心] 添加冗余惩罚辅助损失
        if self.redundancy_weight > 0:
            redundancy_loss = 0.5 * (tf.reduce_mean(raw_red_a) + tf.reduce_mean(raw_red_b))
            self.add_loss(self.redundancy_weight * redundancy_loss)
            self.add_metric(redundancy_loss, name="modal_redundancy_loss", aggregation="mean")

        return {"z_a": z_a, "z_b": z_b, "pair_h": pair_h,
                "att_weights_a": att_a, "att_weights_b": att_b,
                "red_weights_a": red_a, "red_weights_b": red_b,
                "raw_redundancy_a": raw_red_a, "raw_redundancy_b": raw_red_b,
                "blend_alpha": alpha}
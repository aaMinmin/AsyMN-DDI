import pickle
import warnings
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def bitvect_to_numpy(bitvect, dtype=np.float32):
    """
    将 RDKit ExplicitBitVect 转成 numpy array
    """
    arr = np.zeros((bitvect.GetNumBits(),), dtype=dtype)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


@lru_cache(maxsize=16)
def _morgan_generator(radius: int, fp_size: int):
    return GetMorganGenerator(radius=radius, fpSize=fp_size)


def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Morgan（ECFP 风格），默认 radius=2、2048-bit。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    gen = _morgan_generator(radius, n_bits)
    fp = gen.GetFingerprint(mol)
    return bitvect_to_numpy(fp).astype(np.float32)


def get_concat_fingerprint(smiles, morgan_radius=2, n_bits=2048):
    """
    与旧接口兼容；当前仅返回 Morgan 向量。
    """
    return get_morgan_fingerprint(smiles, radius=morgan_radius, n_bits=n_bits)


MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
CONCAT_FP_DIM = MORGAN_NBITS


def smiles_to_concat_fingerprints(smiles, verbose=False):
    """
    单条 SMILES：仅 Morgan 指纹 float32 向量，长度 CONCAT_FP_DIM（2048）。
    异常时返回全零向量。
    """
    try:
        out = get_concat_fingerprint(
            smiles,
            morgan_radius=MORGAN_RADIUS,
            n_bits=MORGAN_NBITS,
        )
        if out.shape[0] != CONCAT_FP_DIM:
            raise ValueError(
                f"concat dim mismatch: got {out.shape[0]}, expect {CONCAT_FP_DIM}"
            )
        return out
    except Exception as e:
        if verbose:
            warnings.warn(f"指纹计算异常 ({smiles}): {e}，返回零向量")
        return np.zeros((CONCAT_FP_DIM,), dtype=np.float32)


class MolFingerprintMLPExtractor(keras.Model):
    """
    从 SMILES 计算 Morgan 指纹（2048-bit），经 MLP 映射到 feature_dim。
    可选 pkl 缓存：dict[str, ndarray]，值为长度 CONCAT_FP_DIM 的向量。
    """

    def __init__(
        self,
        feature_dim=256,
        mlp_hidden=None,
        verbose=False,
        cache_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.verbose = verbose
        self.cache_file = cache_file
        self.fp_cache = None

        if cache_file is not None:
            with open(cache_file, "rb") as f:
                self.fp_cache = pickle.load(f)
            print(
                f"[MolFingerprintMLPExtractor] 已加载指纹缓存: {cache_file} "
                f"({len(self.fp_cache)} 条 SMILES)"
            )

        if mlp_hidden is None:
            mlp_hidden = max(512, self.feature_dim * 2)

        self.mlp_hidden = int(mlp_hidden)
        self.dense1 = layers.Dense(
            self.mlp_hidden,
            activation="relu",
            kernel_regularizer=l2(1e-7),
            name="fp_mlp_dense1",
        )
        self.dropout = layers.Dropout(0.3, name="fp_mlp_dropout")
        self.dense2 = layers.Dense(
            self.feature_dim,
            activation="relu",
            kernel_regularizer=l2(1e-7),
            name="fp_mlp_dense2",
        )
        self.bn = layers.BatchNormalization(name="fp_mlp_bn")

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "feature_dim": self.feature_dim,
                "mlp_hidden": self.mlp_hidden,
                "verbose": self.verbose,
                "cache_file": self.cache_file,
            }
        )
        return cfg

    def call(self, smiles_batch, training=None):
        if smiles_batch.dtype != tf.string:
            raise ValueError("MolFingerprintMLPExtractor 期望 tf.string 类型的 SMILES 批次")

        def process_smiles(smiles_tensor):
            vecs = []
            smiles_np = (
                smiles_tensor.numpy()
                if hasattr(smiles_tensor, "numpy")
                else smiles_tensor
            )
            for s in smiles_np:
                if isinstance(s, bytes):
                    s_str = s.decode("utf-8")
                elif isinstance(s, str):
                    s_str = s
                else:
                    s_str = str(s)

                if self.fp_cache is not None and s_str in self.fp_cache:
                    v = np.asarray(self.fp_cache[s_str], dtype=np.float32).reshape(-1)
                    if v.size != CONCAT_FP_DIM:
                        v = smiles_to_concat_fingerprints(s_str, verbose=self.verbose)
                else:
                    v = smiles_to_concat_fingerprints(s_str, verbose=self.verbose)
                vecs.append(v)
            return np.stack(vecs, axis=0).astype(np.float32)

        raw = tf.py_function(process_smiles, [smiles_batch], tf.float32)
        raw.set_shape([None, CONCAT_FP_DIM])

        x = self.dense1(raw)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn(x, training=training)
        return x

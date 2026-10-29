import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


def rescale_distance_matrix(w): ### For global
    constant_value = tf.constant(1.0,dtype=tf.float32) 
    return (constant_value+tf.math.exp(constant_value))/(constant_value+tf.math.exp(constant_value-w))

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask, structural_bias=None):
    """Scaled dot-product attention with multiplicative SPD modulation.

    structural_bias shape: [batch, heads, query_len, key_len].
    The learnable relation embedding is transformed as W_spd = 1 + tanh(B_spd).
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # Match the supplied SPD reference implementation.
    matmul_qk = tf.nn.relu(matmul_qk)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    if structural_bias is not None:
        structural_weight = 1.0 + tf.math.tanh(
            tf.cast(structural_bias, matmul_qk.dtype)
        )
        scaled_attention_logits = (
            matmul_qk * structural_weight
        ) / tf.math.sqrt(dk)
    else:
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights



class MultiHeadAttention2(keras.layers.Layer):
    def __init__(self, d_model, num_heads,**kwargs):
        super(MultiHeadAttention2, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model,kernel_regularizer=l2(1e-7))
        self.wk = tf.keras.layers.Dense(d_model,kernel_regularizer=l2(1e-7))
        self.wv = tf.keras.layers.Dense(d_model,kernel_regularizer=l2(1e-7))

        self.dense = tf.keras.layers.Dense(d_model,kernel_regularizer=l2(1e-7))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None, structural_bias=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, structural_bias=structural_bias)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation=gelu,kernel_regularizer=l2(1e-7)),
        keras.layers.Dense(d_model,kernel_regularizer=l2(1e-7))
    ])
class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff,rate,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention2(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention2(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    def call(self, x, training=None, encoder_padding_mask=None, structural_bias=None):
      
        x1,x2 = tf.split(x,2,-1)
        x_l,attention_weights_local = self.mha1(
            x1, x1, x1, encoder_padding_mask, structural_bias=structural_bias)
        x_g,attention_weights_global = self.mha2(
            x2, x2, x2, encoder_padding_mask, structural_bias=structural_bias)
        attn_output = tf.concat([x_l,x_g],axis=-1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
       
        x_l_g = out2
        return x_l_g,attention_weights_local,attention_weights_global

class EncoderModel_motif(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size,
                 d_model, num_heads, dff, rate=0.1,
                 max_graph_distance=8, **kwargs):
        super(EncoderModel_motif, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_graph_distance = int(max_graph_distance)
        self.embedding = keras.layers.Embedding(input_vocab_size,
                                                self.d_model)
        # One learnable scalar relation embedding per attention head.
        # IDs 0..K encode motif distances; K+1 encodes global<->motif.
        self.spatial_embedding = keras.layers.Embedding(
            input_dim=self.max_graph_distance + 2,
            output_dim=self.num_heads,
            name="spd_structural_bias")
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)]
        

    def call(self, inputs, spatial_pos=None, training=None):
        # Keras functional usage passes [token_sequence, spatial_pos] as inputs.
        if spatial_pos is None:
            x, spatial_pos = inputs
        else:
            x = inputs

        encoder_padding_mask = create_padding_mask(x)

        # [B, L, L] relation IDs -> [B, heads, L, L] learnable SPD weights.
        spatial_pos = tf.cast(spatial_pos, tf.int32)
        spatial_pos = tf.clip_by_value(
            spatial_pos, 0, self.max_graph_distance + 1)
        structural_bias = self.spatial_embedding(spatial_pos)
        structural_bias = tf.transpose(structural_bias, perm=[0, 3, 1, 2])

        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
        x = self.dropout(x, training = training)
        x_temp = x[:,1:,:]
        x = tf.concat([x[:,0:1,:],x_temp],axis=1)  
        attention_weights_list_local = []
        # xs_local = []
        attention_weights_list_global = []
        for i in range(self.num_layers):
            x,attention_weights_local,attention_weights_global = self.encoder_layers[i](
                x,
                training=training,
                encoder_padding_mask=encoder_padding_mask,
                structural_bias=structural_bias) 
            attention_weights_list_local.append(attention_weights_local) 
            attention_weights_list_global.append(attention_weights_global)
        return x,attention_weights_list_local,attention_weights_list_global,encoder_padding_mask

class Co_Attention_Layer(keras.layers.Layer):
    def __init__(self,graph_feat_size,k,**kwargs):
        self.k = k #128
        self.graph_feat_size = graph_feat_size #256
        super(Co_Attention_Layer, self).__init__(**kwargs)
    def build(self, input_shape):

        self.W_m = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_m',
                                      trainable=True)
        self.W_v = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_v',
                                      trainable=True)
        self.W_q = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_q', 
                                      trainable=True)
        self.W_h = self.add_weight(shape=(1,self.k),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_h',
                                      trainable=True)
        super(Co_Attention_Layer, self).build(input_shape)

    def call(self, inputs): 
        if len(inputs) !=3:
            raise ValueError("Co_Attention_layer requites three inputs")
        #drugA,drugB
        V_n,Q_n = inputs[0],inputs[1]

        V_0 = V_n[:,0,:][:,:,tf.newaxis]
        Q_0 = Q_n[:,0,:][:,:,tf.newaxis] 
        V_r = tf.transpose(V_n[:,1:,:],[0,2,1])
        Q_r = tf.transpose(Q_n[:,1:,:],[0,2,1])
        M_0 = tf.multiply(V_0,Q_0)
        H_v = tf.multiply(tf.tanh(tf.matmul(self.W_v,V_r)),tf.tanh(tf.matmul(self.W_m,M_0)))
        H_q = tf.multiply(tf.tanh(tf.matmul(self.W_q,Q_r)),tf.tanh(tf.matmul(self.W_m,M_0)))
        alpha_v = tf.nn.softmax(tf.matmul(self.W_h,H_v),axis=-1)
        alpha_q = tf.nn.softmax(tf.matmul(self.W_h,H_q),axis=-1)
        vector_v = tf.matmul(alpha_v,tf.transpose(V_r,[0,2,1]))
        vector_q = tf.matmul(alpha_q,tf.transpose(Q_r,[0,2,1]))

        #drugA,drugA_index
        V_n1,Q_n1 = inputs[0],inputs[2]
        V_01 = V_n1[:,0,:][:,:,tf.newaxis]
        Q_01 = Q_n1[:,0,:][:,:,tf.newaxis] 
        V_r1 = tf.transpose(V_n1[:,1:,:],[0,2,1])
        Q_r1 = tf.transpose(Q_n1[:,1:,:],[0,2,1])
        M_01 = tf.multiply(V_01,Q_01)
        H_v1 = tf.multiply(tf.tanh(tf.matmul(self.W_v,V_r1)),tf.tanh(tf.matmul(self.W_m,M_01)))
        H_q1 = tf.multiply(tf.tanh(tf.matmul(self.W_q,Q_r1)),tf.tanh(tf.matmul(self.W_m,M_01)))
        alpha_v1 = tf.nn.softmax(tf.matmul(self.W_h,H_v1),axis=-1)
        alpha_q1 = tf.nn.softmax(tf.matmul(self.W_h,H_q1),axis=-1)
        vector_v1 = tf.matmul(alpha_v1,tf.transpose(V_r1,[0,2,1]))
        vector_q1 = tf.matmul(alpha_q1,tf.transpose(Q_r1,[0,2,1]))

        #drugB,DrugB_index
        V_n2,Q_n2 = inputs[1],inputs[2]
        V_02 = V_n2[:,0,:][:,:,tf.newaxis]
        Q_02 = Q_n2[:,0,:][:,:,tf.newaxis] 
        V_r2 = tf.transpose(V_n2[:,1:,:],[0,2,1])
        Q_r2 = tf.transpose(Q_n2[:,1:,:],[0,2,1])
        M_02 = tf.multiply(V_02,Q_02)
        H_v2 = tf.multiply(tf.tanh(tf.matmul(self.W_v,V_r2)),tf.tanh(tf.matmul(self.W_m,M_02)))
        H_q2 = tf.multiply(tf.tanh(tf.matmul(self.W_q,Q_r2)),tf.tanh(tf.matmul(self.W_m,M_02)))
        alpha_v2 = tf.nn.softmax(tf.matmul(self.W_h,H_v2),axis=-1)
        alpha_q2 = tf.nn.softmax(tf.matmul(self.W_h,H_q2),axis=-1)
        vector_v2 = tf.matmul(alpha_v2,tf.transpose(V_r2,[0,2,1]))
        vector_q2 = tf.matmul(alpha_q2,tf.transpose(Q_r2,[0,2,1]))

        alpha_vall = tf.concat([alpha_v,alpha_v1,alpha_v2],axis=2)
        alpha_qall = tf.concat([alpha_q,alpha_q1,alpha_q2],axis=2)
        vector_vall = tf.concat([vector_v,vector_v1,vector_v2],axis=2)
        vector_qall = tf.concat([vector_q,vector_q1,vector_q2],axis=2)

        return {
            'vector_v':tf.squeeze(vector_vall,1), 
            'vector_q':tf.squeeze(vector_qall,1),
            'alpha_v':alpha_vall,
            'alpha_q':alpha_qall}




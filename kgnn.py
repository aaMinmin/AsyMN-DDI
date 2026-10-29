import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
# from gat import GraphAttentionLayer
from dataset_processed import smiles_to_graph

    
class SelfAttention1(layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(SelfAttention1, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.W_qq = self.add_weight(
            shape=(embedding_dim, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="W_qq1"
        )
        self.W_kk = self.add_weight(
            shape=(embedding_dim, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="W_kk1"
        )
        self.W_vv = self.add_weight(
            shape=(embedding_dim, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="W_vv1"
        )
        
    def call(self, embedinput):
        qq = tf.matmul(embedinput, self.W_qq)
        kk = tf.matmul(embedinput, self.W_kk)
        vv = tf.matmul(embedinput, self.W_vv)
        attention_scores = tf.matmul(qq, kk, transpose_b=True)
        attention_distribution = tf.nn.softmax(attention_scores)
        attention_output = tf.matmul(attention_distribution, vv)
        return attention_output
    
    def get_config(self):
        config = super(SelfAttention1, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention,self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        
        self.W_q = layers.Dense(embedding_dim)
        self.W_k = layers.Dense(embedding_dim)
        self.W_v = layers.Dense(embedding_dim)
        self.W_o = layers.Dense(embedding_dim)

    def call(self, embedinput):
        batch_size = tf.shape(embedinput)[0]
        
        # 生成Q/K/V并分割多头
        q = self.W_q(embedinput)  # (B, L, D)
        k = self.W_k(embedinput)
        v = self.W_v(embedinput)
        
        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))  # (B, L, H, d)
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (B, H, L, d)
        
        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.head_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # 注意力计算
        attn_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        
        output = tf.matmul(attn_weights, v)  # (B, H, L, d)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, L, H, d)
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.head_dim))
        return self.W_o(output)
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
        })
        return config

class GNN1(keras.Model):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name,preset_initializer=None,**kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.kg = dataset["dataset1"]
        self.dict1 = dict1
        self.drug_name = drug_name
        self.args = args
        
        if preset_initializer is None:
            # 默认使用随机初始化（可改为其他初始化方式）
            preset_initializer = tf.keras.initializers.RandomNormal()
        self.preset_matrix = self.add_weight(
            shape=(2459, args.embedding_num),
            initializer=preset_initializer,
            trainable=True,  # 设为False可冻结该参数
            name="preset_matrix"
        )

        self.drug_embed = layers.Embedding(
            input_dim=2459, 
            output_dim=args.embedding_num, 
            name = "dr_gnn1"
        )
        self.rela_embed = layers.Embedding(
            input_dim=relation_len["dataset1"], 
            output_dim=args.embedding_num, 
            name = "rel_gnn1"
        )
        self.ent_embed = layers.Embedding(
            input_dim=tail_len["dataset1"], 
            output_dim=args.embedding_num, 
            name = "ent_gnn1"
        )
        
        self.shared_W1 = self.add_weight(
            shape=(args.embedding_num, args.embedding_num),
            initializer="RandomNormal",
            trainable=True,
            name="gnn1_W1"
        )
        self.shared_b1 = self.add_weight(
            shape=(args.embedding_num,),
            initializer="RandomNormal",
            trainable=True,
            name="gnn1_b1"
        )

        self.mh_attention = MultiHeadSelfAttention(args.embedding_num, name="multi_head")
        self.self_attention1 = SelfAttention1(args.embedding_num, name="self_attention_gnn1")
        # self.linear2 = Sequential([
        #     layers.Dropout(0.3)
        #     #layers.BatchNormalization()
        # ])
        self.linear1 = Sequential([
            layers.Dense(args.embedding_num, activation='relu', kernel_regularizer=l2(1e-7)),
            layers.Dropout(0.3)
            #layers.BatchNormalization()
        ])
        
        # self.linear2 = Sequential([
        #     layers.Dense(args.embedding_num*2, activation='relu', kernel_regularizer=l2(1e-7)),
        #     #layers.BatchNormalization()
        # ])

        # self.smiles_gnn = SMILESGNN(args)
        
    def call(self, idx1,idx2):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = tf.convert_to_tensor(self.drug_name, dtype=tf.float32)
        adj_tail = tf.convert_to_tensor(adj_tail, dtype=tf.float32)
        adj_relation = tf.convert_to_tensor(adj_relation, dtype=tf.float32)
        
        drug_embedding = self.drug_embed(drug_name)+self.preset_matrix
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        
        # # 应用自注意力机制
        rela_embedding2 = self.self_attention1(rela_embedding)
        rela_embedding2 = self.mh_attention(rela_embedding2) + rela_embedding2
        # # rela_embedding2 = self.linear2(rela_embedding2)
        ent_embedding2 = self.self_attention1(ent_embedding)
        ent_embedding2 = self.mh_attention(ent_embedding2)+ ent_embedding2
        # # ent_embedding2 = self.linear2(ent_embedding2)

        
        reshape_drugembed = tf.reshape(drug_embedding, (2459, 1, args.embedding_num))
        drug_rel = tf.multiply(reshape_drugembed, rela_embedding2)
        drug_rel_weigh = tf.matmul(drug_rel, self.shared_W1) + self.shared_b1
        drug_rel_score = tf.reduce_sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = tf.nn.softmax(drug_rel_score)
        
        reshape_drgresc = tf.reshape(drug_rel_score, (2459, 1, args.neighbor_sample_size))
        weighted_ent = tf.matmul(reshape_drgresc, ent_embedding2)
        weighted_ent = tf.reshape(weighted_ent, (2459, args.embedding_num))
        drug_embedding = tf.reshape(drug_embedding, (2459, args.embedding_num))
        drug_e = tf.concat([weighted_ent, drug_embedding,self.preset_matrix], axis=-1)
        drug_f = self.linear1(drug_e)


        # seqa = self.smiles_gnn(atom1,dist1)
        # seqb = self.smiles_gnn(atom2,dist2)
        # print(atom1.shape)
        # print(atom2.shape)
        # seqa_f = self.linear2(seqa)
        # seqb_f = self.linear2(seqb)
        # # 获取 SMILES 序列并计算特征
        # smiles_features = []
        # for drug_id in range(2459):
        #     smiles = self.smiles_dict.get(drug_id, None)
        #     if smiles is not None:
        #         atoms, adj_matrix = smiles_to_graph(smiles)
        #         atoms = tf.convert_to_tensor(atoms, dtype=tf.int32)
        #         adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        #         smiles_feature = self.smiles_gnn(atoms, adj_matrix)
        #         smiles_features.append(smiles_feature)
        #     else:
        #         smiles_features.append(tf.zeros((args.embedding_num,), dtype=tf.float32))
        
      #  smiles_features = tf.stack(smiles_features)
        # drug_f = tf.concat([drug_f, seqa_f, seqb_f], axis=-1)
        
        return drug_f, idx1,idx2

    def arrge(self, kg, drug_name_id, neighbor_sample_size,n_drug=2459):
       # n_drug = 2459
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.float32)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.float32)
        for i in drug_name_id: #i是smiles序列
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
              #  replace= n_neighbor < neighbor_sample_size
                replace=False if n_neighbor>= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[j][0] for j in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[j][1] for j in sample_indices])
        return adj_tail, adj_relation
    
    '''def get_config(self):
        config = super(GNN1, self).get_config()
        config.update({
            'dataset': self.kg,
            'tail_len': {'dataset1': self.ent_embed.input_dim},
            'relation_len': {'dataset1': self.rela_embed.input_dim},
            'args': self.args,
            'dict1': self.dict1,
            'drug_name': self.drug_name,
        })
        return config'''
    

class GNN2(keras.Model):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name,preset_initializer=None,**kwargs):
        super(GNN2, self).__init__(**kwargs)
        self.kg = dataset["dataset2"]
        self.dict1 = dict1
        self.drug_name = drug_name
        self.args = args
        
        if preset_initializer is None:
            # 默认使用随机初始化（可改为其他初始化方式）
            preset_initializer = tf.keras.initializers.RandomNormal()
        self.preset_matrix = self.add_weight(
            shape=(2459, args.embedding_num),
            initializer=preset_initializer,
            trainable=True,  # 设为False可冻结该参数
            name="preset_matrix"
        )

        self.drug_embed = layers.Embedding(
            input_dim=2459, 
            output_dim=args.embedding_num, 
            name = "dr_gnn1"
        )
        self.rela_embed = layers.Embedding(
            input_dim=relation_len["dataset2"], 
            output_dim=args.embedding_num, 
            name = "rel_gnn2"
        )
        self.ent_embed = layers.Embedding(
            input_dim=tail_len["dataset2"], 
            output_dim=args.embedding_num, 
            name = "ent_gnn2"
        )

        self.shared_W2 = self.add_weight(
            shape=(args.embedding_num, args.embedding_num),
            initializer="RandomNormal",
            trainable=True,
            name="gnn2_W2"
        )
        self.shared_b2 = self.add_weight(
            shape=(args.embedding_num,),
            initializer="RandomNormal",
            trainable=True,
            name="gnn2_b2"
        )

        self.mh_attention2 = MultiHeadSelfAttention(args.embedding_num, name="multi_head2")
        self.self_attention2 = SelfAttention1(args.embedding_num, name="self_attention_gnn2")  # 添加自注意力层
        # self.linear2 = Sequential([
        #     layers.Dropout(0.3)
        #     #layers.BatchNormalization()
        # ])
        self.linear1 = Sequential([
            layers.Dense(args.embedding_num, activation='relu', kernel_regularizer=l2(1e-7)),
            layers.Dropout(0.3)
          #  layers.BatchNormalization()
        ])

        # self. smiles_gnn = SMILESGNN(args)

    def call(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn1_embedding, idx1,idx2 = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)

        drug_name = tf.convert_to_tensor(self.drug_name, dtype=tf.float32)
        adj_tail = tf.convert_to_tensor(adj_tail, dtype=tf.float32)
        adj_relation = tf.convert_to_tensor(adj_relation, dtype=tf.float32)
        
        drug_embedding = self.drug_embed(drug_name)+self.preset_matrix
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        
        # # self-attention
        rela_embedding2 = self.self_attention2(rela_embedding)
        rela_embedding2 = self.mh_attention2(rela_embedding2) + rela_embedding2
        ent_embedding2 = self.self_attention2(ent_embedding)
        ent_embedding2 = self.mh_attention2(ent_embedding2)+ ent_embedding2
        
        reshape_drugembed = tf.reshape(drug_embedding, (2459, 1, args.embedding_num))
        drug_rel = tf.multiply(reshape_drugembed, ent_embedding2)
        drug_rel_weigh = tf.matmul(drug_rel, self.shared_W2) + self.shared_b2
        drug_rel_score = tf.reduce_sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = tf.nn.softmax(drug_rel_score)
        
        reshape_drgresc = tf.reshape(drug_rel_score, (2459, 1, args.neighbor_sample_size))
        weighted_ent = tf.matmul(reshape_drgresc, rela_embedding2)
        weighted_ent = tf.reshape(weighted_ent, (2459, args.embedding_num))
        drug_embedding = tf.reshape(drug_embedding, (2459, args.embedding_num))
        drug_e = tf.concat([weighted_ent, drug_embedding,self.preset_matrix], axis=-1)
        drug_f = self.linear1(drug_e)
        
        return drug_f, gnn1_embedding, idx1,idx2

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=2459):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.float32)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.float32)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor>= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[j][0] for j in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[j][1] for j in sample_indices])
        return adj_tail, adj_relation
    
class FusionLayer(keras.Model):
    """
    药物级 KG 表征 + 药物对 kg_pair。
    返回 (kg_a, kg_b, kg_pair)，供 ExpertMultimodalFusionEntropy 层 1/2 使用。
    """

    def __init__(self, args):
        super(FusionLayer, self).__init__()
        d = args.embedding_num
        reg = l2(1e-7)
        self.kg_a_proj = keras.Sequential(
            [
                layers.Dense(d, activation="relu", kernel_regularizer=reg, name="kg_a_dense"),
                layers.Dropout(0.3),
            ],
            name="kg_a_proj",
        )
        self.kg_b_proj = keras.Sequential(
            [
                layers.Dense(d, activation="relu", kernel_regularizer=reg, name="kg_b_dense"),
                layers.Dropout(0.3),
            ],
            name="kg_b_proj",
        )
        self.kg_pair_proj = keras.Sequential(
            [
                layers.Dense(d, activation="relu", kernel_regularizer=reg, name="kg_pair_dense"),
                layers.Dropout(0.3),
            ],
            name="kg_pair_proj",
        )

    def call(self, arguments):
        gnn2_embedding, gnn1_embedding, idx1, idx2 = arguments
        drugA_indices = tf.cast(idx1, tf.int64)
        drugB_indices = tf.cast(idx2, tf.int64)

        gnn1_embedding_A = tf.gather(gnn1_embedding, drugA_indices)
        gnn2_embedding_A = tf.gather(gnn2_embedding, drugA_indices)
        gnn1_embedding_B = tf.gather(gnn1_embedding, drugB_indices)
        gnn2_embedding_B = tf.gather(gnn2_embedding, drugB_indices)

        kg_a_in = tf.concat([gnn1_embedding_A, gnn2_embedding_A], axis=-1)
        kg_b_in = tf.concat([gnn1_embedding_B, gnn2_embedding_B], axis=-1)
        kg_a = self.kg_a_proj(kg_a_in)
        kg_b = self.kg_b_proj(kg_b_in)
        kg_pair_in = tf.concat(
            [kg_a, kg_b, tf.abs(kg_a - kg_b), kg_a * kg_b],
            axis=-1,
        )
        kg_pair = self.kg_pair_proj(kg_pair_in)
        return kg_a, kg_b, kg_pair
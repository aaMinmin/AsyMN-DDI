import pandas as pd
import numpy as np
import tensorflow as tf
import networkx as nx
from rdkit import Chem
from utils import *

# Must match EncoderModel_motif(max_graph_distance=8).
SPD_MAX_GRAPH_DISTANCE = 8


def build_spd_relation_matrix(dist_matrix, max_graph_distance=SPD_MAX_GRAPH_DISTANCE):
    """Convert motif shortest-path distances to categorical SPD relation IDs.

    Input excludes <global> and has shape [N, N]. Output includes <global>
    and has shape [N+1, N+1]. IDs 0..K encode motif distances; K+1 encodes
    global<->motif relations.
    """
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square [N, N] matrix")

    k = int(max_graph_distance)
    clipped = np.nan_to_num(dist_matrix, nan=k, posinf=k, neginf=0)
    clipped = np.clip(clipped, 0, k).astype(np.int32)

    n = clipped.shape[0]
    spatial_pos = np.zeros((n + 1, n + 1), dtype=np.int32)
    spatial_pos[1:, 1:] = clipped
    global_relation_id = k + 1
    spatial_pos[0, 1:] = global_relation_id
    spatial_pos[1:, 0] = global_relation_id
    spatial_pos[0, 0] = 0
    return spatial_pos


def _motif_distance_matrix(entry, num_nodes):
    """Use precomputed motif SPD; fall back to shortest paths over motif edges."""
    dist_matrix = entry.get("dist_matrix")
    if dist_matrix is not None:
        dist_matrix = np.asarray(dist_matrix)
        if dist_matrix.shape == (num_nodes, num_nodes):
            return dist_matrix

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(entry.get("edges", []))
    dist_matrix = np.full((num_nodes, num_nodes), 1e9, dtype=np.float32)
    np.fill_diagonal(dist_matrix, 0.0)
    for source, distances in nx.all_pairs_shortest_path_length(graph):
        for target, distance in distances.items():
            dist_matrix[source, target] = float(distance)
    return dist_matrix


def get_adj_matrix(num_list,edges):#生成一个邻接矩阵（adjacency matrix），表示图中节点（原子）之间的连接关系
    adjoin_matrix = np.eye(len(num_list))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return adjoin_matrix#返回：一个距离矩阵，矩阵中的元素表示节点之间的最短距离

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 获取原子和键的信息
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    # 构建邻接矩阵
    adj_matrix = np.zeros((len(atoms), len(atoms)), dtype=np.float32)
    for bond in bonds:
        adj_matrix[bond[0], bond[1]] = 1
        adj_matrix[bond[1], bond[0]] = 1

    return atoms, adj_matrix

def get_dist_matrix(num_list,edges):#生成一个距离矩阵（distance matrix），表示图中节点之间的最短路径距离
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((len(num_list),len(num_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row,col] = 0
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(),key = lambda x:x[0]))
        temp_node_dist_dict = {i:node_relative_distance.get(i) if \
        node_relative_distance.get(i)!= None else 1e9 for i in graph_nodes} ### Refer to rdkit Chem.GetDistanceMatrix(mol)
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[dist][graph_nodes] =  temp_node_dist_list
    return dist_matrix.astype(np.float32)#返回一个字典，包含多个特征，例如输入原子特征、邻接矩阵、距离矩阵等

def molgraph_rep(smi,cliques):#生成分子图的表示，包括邻接矩阵、距离矩阵和原子特征等。
    def atom_to_motif_match(atom_order,cliques):#生成一个矩阵，表示原子与团之间的匹配关系。
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques),len(atom_order)))
        for th,cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th,atom_order.index(i)] = 1
        return temp_matrix#返回：一个矩阵，行表示团，列表示原子，元素为1表示原子属于该团。

    #功能：获取分子图的邻接矩阵和距离矩阵。
    def get_adj_dist_matrix(mol_graph,smi):#一个分子图 一个smiles序列

        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms()
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix,dist_matrix #返回邻接矩阵和距离矩阵
    single_dict = {'input_atom_features':[],
            'atom_match_matrix':[],
            'sum_atoms':[],
            'adj_matrix':[],
            'dist_matrix':[]
            }
    array_rep = array_rep_from_smiles(smi)#包含分子图所有必要信息的数组表示
    summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
    atom_features = array_rep['atom_features']
    all_bond_features = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
        if len(atom_neighbors_list)==0:
                true_summed_degree = np.zeros((atom_features.shape[0], 10),'float32')
        else:
                true_summed_degree = bond_features_by_degree(atom_features.shape[0],summed_degrees,degree)
        all_bond_features.append(true_summed_degree)
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'],cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'],axis=1),(-1,1))
    out_bond_features = 0
    for arr in all_bond_features:
        out_bond_features = out_bond_features + arr
    single_dict['input_atom_features'] = np.concatenate([atom_features,out_bond_features],axis=1)
    adj_matrix,dist_matrix = get_adj_dist_matrix(array_rep,smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix
    single_dict = {key:np.array(value,dtype='float32') for key,value in single_dict.items()}
    return single_dict


class Graph_Bert_Dataset_fine_tune(object):
    def __init__(self,train_set,val_set,tst_set,dict,tokenizer,batch_size,map_dict,label_field='DDI'):
        #接收训练、验证、测试集
        self.train_set  = train_set
        self.val_set = val_set
        self.tst_set = tst_set
        self.dict = dict
        self.train_set[label_field] = self.train_set[label_field].map(int)
        self.val_set[label_field] = self.val_set[label_field].map(int)
        self.tst_set[label_field] = self.tst_set[label_field].map(int)
        self.label_field = label_field#标签字段
        self.batch_size = batch_size#批次大小
        self.tokenizer = tokenizer#分词器
        self.pad_value = self.tokenizer.vocab['<pad>'] 
        self.map_dict = map_dict#映射字典

    def get_data(self):
        
       
        #get train drugA
        drugA=[]
        drugB=[]
        lable_relation=[]
        for i in self.train_set['drug_A']:
            drugA.append(i)
        for j in self.train_set['drug_B']:
            drugB.append(j)
        for z in self.train_set[self.label_field]:
            lable_relation.append(z)
      #  drug_name=[dict[name] for name in dict] #drug_name是所有的药物.
        drugA_id = [self.dict[i] for i in drugA]
        drugB_id = [self.dict[i] for i in drugB]
        tr_datasetsA = {"drugA1": drugA_id} 
        tr_datasetsB = {"drugB1": drugB_id}
        tr_datasetsA = pd.DataFrame(data=tr_datasetsA)
        tr_datasetsB = pd.DataFrame(data=tr_datasetsB)
        tr_datasetsA = tr_datasetsA.to_numpy()
        tr_datasetsB = tr_datasetsB.to_numpy()

        #get val drugA
        drugA1=[]
        drugB1=[]
        lable_relation=[]
        for i in self.val_set['drug_A']:
            drugA1.append(i)
        for j in self.val_set['drug_B']:
            drugB1.append(j)
        for z in self.val_set[self.label_field]:
            lable_relation.append(z)
      #  drug_name=[dict[name] for name in dict] #drug_name是所有的药物.
        drugA1_id = [self.dict[i] for i in drugA1]
        drugB1_id = [self.dict[i] for i in drugB1]
        val_datasetsA = {"drugA1": drugA1_id}
        val_datasetsB = {"drugB1": drugB1_id}
        val_datasetsA = pd.DataFrame(data=val_datasetsA)
        val_datasetsB = pd.DataFrame(data=val_datasetsB)
        val_datasetsA = val_datasetsA.to_numpy()
        val_datasetsB = val_datasetsB.to_numpy()

        #get test drugA
        drugA2=[]
        drugB2=[]
        lable_relation=[]
        for i in self.tst_set['drug_A']:
            drugA2.append(i)
        for j in self.tst_set['drug_B']:
            drugB2.append(j)
        for z in self.tst_set[self.label_field]:
            lable_relation.append(z)
      #  drug_name=[dict[name] for name in dict] #drug_name是所有的药物.
        drugA2_id = [self.dict[i] for i in drugA2]
        drugB2_id = [self.dict[i] for i in drugB2]
        test_datasetsA = {"drugA1": drugA2_id}
        test_datasetsB = {"drugB1": drugB2_id}
        test_datasetsA = pd.DataFrame(data=test_datasetsA)
        test_datasetsB = pd.DataFrame(data=test_datasetsB)
        test_datasetsA = test_datasetsA.to_numpy()
        test_datasetsB = test_datasetsB.to_numpy()

        #train
        self.dataset1 = tf.data.Dataset.from_tensor_slices((
            self.train_set['drug_A'], 
            self.train_set['drug_B'],
            tr_datasetsA,
            tr_datasetsB,
            self.train_set[self.label_field]
            ))


        self.dataset1 = self.dataset1.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),      # x1
                tf.constant(0, dtype=tf.int64),      # x2
                tf.constant(0, dtype=tf.int64),      # datasetsA
                tf.constant(0, dtype=tf.int64),      # datasetsB
                tf.constant(0, dtype=tf.int32),      # spatial_pos1
                tf.constant(0, dtype=tf.int32),      # spatial_pos2
                tf.constant('', dtype=tf.string),    # smiles1
                tf.constant('', dtype=tf.string),    # smiles2
                tf.constant(0, dtype=tf.int64),      # y
            ),
            padded_shapes=(
                tf.TensorShape([None]),              # x1
                tf.TensorShape([None]),              # x2
                tf.TensorShape([None]),              # datasetsA
                tf.TensorShape([None]),              # datasetsB
                tf.TensorShape([None, None]),        # spatial_pos1
                tf.TensorShape([None, None]),        # spatial_pos2
                tf.TensorShape([]),                  # smiles1 (标量字符串)
                tf.TensorShape([]),                  # smiles2
                tf.TensorShape([1]),                 # y
            ),
        ).prefetch(20)
        
        #val
        self.dataset2 = tf.data.Dataset.from_tensor_slices((
            self.val_set['drug_A'], 
            self.val_set['drug_B'],
            val_datasetsA,
            val_datasetsB,
            self.val_set[self.label_field]))


        self.dataset2 = self.dataset2.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                tf.constant('', dtype=tf.string),
                tf.constant('', dtype=tf.string),
                tf.constant(0, dtype=tf.int64),
            ),
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([1]),
            ),
        ).prefetch(20)
        
        #test
        self.dataset3 = tf.data.Dataset.from_tensor_slices((self.tst_set['drug_A'],
                                                             self.tst_set['drug_B'],
                                                             test_datasetsA,
                                                             test_datasetsB,
                                                             self.tst_set[self.label_field]))

        self.dataset3 = self.dataset3.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                tf.constant('', dtype=tf.string),
                tf.constant('', dtype=tf.string),
                tf.constant(0, dtype=tf.int64),
            ),
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([1]),
            ),
        ).prefetch(20)
        return self.dataset1, self.dataset2,self.dataset3 #训练 验证 测试数据.

  
    def numerical_seq(self, smiles1, smiles2, datasetsA, datasetsB, labels):
        """Return motif tokens, KG IDs, SPD matrices, SMILES strings, and label."""
        smiles1 = smiles1.numpy().decode()
        smiles2 = smiles2.numpy().decode()

        entry1 = self.map_dict[smiles1]
        entry2 = self.map_dict[smiles2]
        nums_list1 = [self.tokenizer.vocab['<global>']] + entry1['nums_list']
        nums_list2 = [self.tokenizer.vocab['<global>']] + entry2['nums_list']

        # processing.py constructs these distances over motif nodes and edges.
        dist1 = _motif_distance_matrix(entry1, len(nums_list1) - 1)
        dist2 = _motif_distance_matrix(entry2, len(nums_list2) - 1)
        spatial_pos1 = build_spd_relation_matrix(dist1, SPD_MAX_GRAPH_DISTANCE)
        spatial_pos2 = build_spd_relation_matrix(dist2, SPD_MAX_GRAPH_DISTANCE)

        if len(nums_list1) != spatial_pos1.shape[0]:
            raise ValueError(
                f"Drug A token/SPD mismatch: {len(nums_list1)} tokens vs {spatial_pos1.shape}"
            )
        if len(nums_list2) != spatial_pos2.shape[0]:
            raise ValueError(
                f"Drug B token/SPD mismatch: {len(nums_list2)} tokens vs {spatial_pos2.shape}"
            )

        y = np.array([labels]).astype('int64')
        x1 = np.asarray(nums_list1, dtype='int64')
        x2 = np.asarray(nums_list2, dtype='int64')
        return (
            x1, x2, datasetsA, datasetsB,
            spatial_pos1.astype('int32'), spatial_pos2.astype('int32'),
            np.string_(smiles1), np.string_(smiles2), y
        )

    def tf_numerical_seq(self, smiles1, smiles2, datasetsA, datasetsB, labels):
        x1, x2, datasetsA, datasetsB, spatial_pos1, spatial_pos2, sm1, sm2, y = tf.py_function(
            self.numerical_seq,
            [smiles1, smiles2, datasetsA, datasetsB, labels],
            [tf.int64, tf.int64, tf.int64, tf.int64,
             tf.int32, tf.int32, tf.string, tf.string, tf.int64]
        )
        x1.set_shape([None])
        x2.set_shape([None])
        datasetsA.set_shape([None])
        datasetsB.set_shape([None])
        spatial_pos1.set_shape([None, None])
        spatial_pos2.set_shape([None, None])
        sm1.set_shape([])
        sm2.set_shape([])
        y.set_shape([None])
        return x1, x2, datasetsA, datasetsB, spatial_pos1, spatial_pos2, sm1, sm2, y

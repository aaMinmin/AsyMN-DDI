import warnings
warnings.warn('ignore')
import numpy as np
from rdkit.Chem import MolFromSmiles,MolToSmiles
from data_processing.features import (atom_features,bond_features,
                        one_of_k_encoding, one_of_k_encoding_unk)

degrees = [0, 1, 2, 3, 4, 5]
class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):#用于合并两个分子图
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))# gain node

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees} 
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self,ntype,features,rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)
    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph#返回包含所有子图信息的 big_graph 实例

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(MolToSmiles(mol)) 
    if not mol:#分子对象是否有效
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():#原子
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():#键
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph#将一个 SMILES 字符串转换成一个分子图对象，这个对象包含了分子的原子、化学键以及它们之间的连接关系

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)#使用 SMILES 字符串创建一个分子图对象
    molgraph = MolGraph()#创建MolGraph实例
    molgraph.add_subgraph(graph)#将分子图对象添加到MolGraph实例中。MolGraph是一个专门用于处理分子图的类
    molgraph.sort_nodes_by_degree('atom')#根据原子的度数，即连接的键的数量，对分子图中的原子节点进行排序，有助于后续的数据处理和特征提取。
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),#原子特征数组
                'bond_features' : molgraph.feature_array('bond'),#键特征数组
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.#原子的邻居列表
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only. #RDKit 索引数组，用于绘图和后续分析
    for degree in degrees:#循环不同的度数，提取每个原子的邻居信息
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep#返回包含分子图所有必要信息的数组表示

if __name__ == "__main__":
    # smiles="CC1CCC2=CC(F)=CC3=C2N1C=C(C(O)=O)C3=O"
    smiles="NC1=CC=CC=C1"
    arrayrep = array_rep_from_smiles(smiles)
    # print(arrayrep)

import copy
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from data_processing.mol_graph import array_rep_from_smiles
import xml.etree.ElementTree as ET
from IPython.display import SVG, display
from matplotlib import colors
import rdkit.Chem as Chem
import json
from rdkit import DataStructs
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
import os

class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value:key for key,value in self.vocab.items()}
    def tokenize(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}") 
            return None,None,None
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))#转换成分子对象
        ids,edge = self.tree_decomp(mol) #进行树分解
        #ids是所有团的列表，每个团是一个原子索引的列表；
        # edge包含团之间的连接关系，通常是一个包含边的元组列表，这些边定义了团之间的连接
        motif_list = []
        for id_ in ids:
            _,token_mols = self.get_clique_mol(mol,id_)#从一个较大的分子对象 mol 中提取出一个较小的分子片段（团，clique）
            #_是规范化后的团分子对象；token_mols是返回的团字符串smiles
            token_id = self.vocab.get(token_mols)#有团字符串的话，就赋予token_id;不在就none.
            if token_id!=None:#token id不为空
                motif_list.append(token_id) #喂入token_id
            else: 
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list,edge,ids #团字符串、团之间的边,所有团的列表（团里面包含原子）
    def sanitize(self,mol):
        try:
            smiles = self.get_smiles(mol)#将分子对象 mol 转换为SMILES字符串
            mol = self.get_mol(smiles)#再从这个SMILES字符串重新创建一个分子对象
        except Exception as e:
            return None
        return mol
    def get_mol(self,smiles):
         #使用 Chem.MolFromSmiles(smiles) 尝试从输入的SMILES字符串 smiles 创建一个RDKit分子对象 mol
        mol = Chem.MolFromSmiles(smiles) 
        if mol is None:
            return None
        Chem.Kekulize(mol)#如果mol不是none，则使用kekulize函数对分子对象进行kekul化（将分子中的芳香环转换为具有交替双键和单键的非芳香环的过程）
        return mol
    def get_smiles(self,mol):
        #将一个 RDKit 分子对象 mol 转换回 SMILES 字符串
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    def get_clique_mol(self,mol,atoms_ids):
    # get the fragment of clique
    #输入：self（表示类的实例），mol（一个 RDKit 分子对象），和 atoms_ids（一个包含原子 ID 的列表，这些原子 ID 定义了要提取的子结构）
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False) #将分子对象mol中由atoms——ids指定的原子及其连接的原子转换为SMILES字符串。
        #kekuleSmiles=False意思是生成SMILES字符串时不进行kekulization
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)#将上面的smiles转换为rdkit分子对象new_mol.；sanitize=False不进行分子规范化处理
        new_mol = self.copy_edit_mol(new_mol).GetMol()#复制和编辑分子对象 new_mol，并返回编辑后的分子对象
        new_mol = self.sanitize(new_mol)  # We assume this is not None
        return new_mol,smiles
    def copy_atom(self,atom):#拷贝原子
        new_atom = Chem.Atom(atom.GetSymbol())#通过调用 atom.GetSymbol() 方法获取 atom 的元素符号，并用它来初始化新的原子对象
        new_atom.SetFormalCharge(atom.GetFormalCharge())#设置一样的电荷。
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())#设置一样的映射编号
        return new_atom
    def copy_edit_mol(self,mol):#拷贝和编辑原子复制一个 RDKit 分子对象中的所有原子和键，创建一个新的分子对象
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))#创建一个新的可编辑的rdkit分子对象new_mol（Chem.MolFromSmiles('')是创建一个空的分子对象）
        for atom in mol.GetAtoms():#遍历原始分子对象mol中的所有原子
            new_atom = self.copy_atom(atom)#对每个原子都进行拷贝
            new_mol.AddAtom(new_atom)#将新创建的原子new_atom添加到可编辑分子对象new_mol中

        for bond in mol.GetBonds():#遍历原始分子对象 mol 中的所有键
            a1 = bond.GetBeginAtom().GetIdx()#获取当前键的起始原子的索引
            a2 = bond.GetEndAtom().GetIdx()#获取当前键的结束原子的索引
            bt = bond.GetBondType()#获取当前键的类型
            new_mol.AddBond(a1, a2, bt)#在new_mol中添加新键，连接索引为 a1 和 a2 的原子，键类型为 bt。
        return new_mol#返回新创建的分子对象 new_mol


    def tree_decomp(self,mol):#将分子结构分解成一个树形结构，其中每个节点代表一个团，每条边代表团之间的关系。
        n_atoms = mol.GetNumAtoms()#获取原子的数量
        if n_atoms == 1:#如果为1的话
            return [[0]], []#返回一个包含该原子索引的团和一个空的边列表。

        cliques = []#初始化一个空列表，用于存储团
        for bond in mol.GetBonds():#遍历分子中所有的键
            a1 = bond.GetBeginAtom().GetIdx()#获取键的起始原子索引
            a2 = bond.GetEndAtom().GetIdx()#获取键的结束原子索引
            if not bond.IsInRing():#检查每个键是否属于任何一个环结构，要是键不在环中，那么将创建一个包含着两个原子索引的团（clique）
                cliques.append([a1, a2])

        # get rings
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]#获取分子中的对称环结构（SymmSSSR），并将它们转换为列表。
        cliques.extend(ssr)#将环结构添加到团列表中

        nei_list = [[] for i in range(n_atoms)] #初始化一个列表，用于存储每个原子相邻的团索引
        for i in range(len(cliques)):#对每个团都进行循环（第几个团）
            for atom in cliques[i]:#遍历团的每个原子
                nei_list[atom].append(i)#将团索引添加到对应原子的相邻团列表中

        # Merge Rings with intersection > 2 atoms
        for i in range(len(cliques)):#合并那些相交超过两个原子的环状团，这是树分解过程中的一个步骤。
            #用于确保团之间正确地表示分子中的连接关系
            if len(cliques[i]) <= 2: continue #一个团的原子小于2， 就跳过。因为单个原子或者两个原子
            for atom in cliques[i]: #团的原子大于2的团。 对每个原子进行遍历
                for j in nei_list[atom]:#遍历团索引
                    if i >= j or len(cliques[j]) <= 2: continue#如果团 j 的索引小于或等于团 i 的索引（为了避免重复处理相同的团对），或者团 j 中的原子数量小于或等于两个，跳过这个团。
                    inter = set(cliques[i]) & set(cliques[j])# 计算团 i 和团 j 的交集，即这两个团共有的原子集合。
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])#将团 j 中的所有原子添加到团 i 中
                        cliques[i] = list(set(cliques[i]))#将团 i 转换为一个不包含重复元素的列表，确保团中的原子是唯一的。
                        cliques[j] = []#清空团 j，因为它已经被合并到团 i 中了

#更新团列表和每个原子对应的相邻团索引列表。这个操作跟合并团之前是一样的，这是确保团列表中只包含有效的团。
        cliques = [c for c in cliques if len(c) > 0]#用于过滤掉空的团（当两个团合并的时候，较小的团会被清空）
        nei_list = [[] for i in range(n_atoms)]#初始化一个新的列表 nei_list，它的长度等于分子中原子的数量。每个元素是一个空列表，用于存储每个原子相邻的团的索引
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)#初始化一个默认值为整数类型的字典 edges，用于存储团之间的边和它们的权重。
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]#获取原子 atom 所属的所有团的索引列表。
            bonds = [c for c in cnei if len(cliques[c]) == 2]#从 cnei 中筛选出只包含两个原子的团（即单键）。
            rings = [c for c in cnei if len(cliques[c]) > 4]#从 cnei 中筛选出包含多于四个原子的团（即复杂环）。
            if len(bonds) > 2 or (len(bonds) == 2 and len(
                    cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:  # Multiple (n>2) complex rings
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        #计算团之间的最大生成树
        row, col, data = zip(*edges)#将 edges 字典中的键值对解包，edges 字典的键是元组 (c1, c2) 表示两个团之间的边，值是边的权重。这行代码将边的起点、终点和权重分别打包成三个列表。
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()#获取最大生成树中非零元素的行和列索引，这些非零元素代表最大生成树中的边。
        edges = [(row[i], col[i]) for i in range(len(row))]#根据非零元素的行和列索引构建最大生成树的边列表，每个元组 (row[i], col[i]) 表示一条边
        return (cliques, edges)#返回团列表和最大生成树的边列表

class GroundTruthSub(): ### To visualize substructures described in literature
    def __init__(self,smiles,atoms_to_highlight,colors = '#c44240'):
        self.smiles = smiles 
        self.atoms_to_highlight = atoms_to_highlight
        self.colors = colors
        self.bond_diaplay = 'fill:{};fill-rule:evenodd;stroke:{};stroke-width:3px;stroke-linejoin:miter;stroke-opacity:1.0'
        self.atom_display = 'fill:{};fill-rule:evenodd;stroke:{};stroke-width:0.1px;stroke-linejoin:miter;stroke-opacity:1.0'
        self.svg_init,self.mol = self.init_svg() 
        self.bonds_to_highlight = self.init_bonds_to_highlight() 
    def init_svg(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        d = rdMolDraw2D.MolDraw2DSVG(250, 250)
        d.SetFontSize(0.68)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        hcolor = colors.to_rgb('white')
        d.drawOptions().setHighlightColour(hcolor)
        d.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)}) 
        d.drawOptions().highlightRadius=.10
        d.DrawMolecule(tm,highlightAtoms=self.atoms_to_highlight)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        svg = svg.replace('svg:','')
        return svg,mol 
    def init_bonds_to_highlight(self):
        all_bond = set()
        for i in self.atoms_to_highlight:
            for j in self.atoms_to_highlight:
                if i!=j:
                    b = self.mol.GetBondBetweenAtoms(i,j)
                    if b!=None:
                        all_bond.add(b.GetIdx())
        return list(all_bond)
    def visualize(self):
        parsed = ET.fromstring(self.svg_init)
        new_parsed = copy.deepcopy(parsed)
        for i,j in enumerate(parsed):
            if j.tag == '{http://www.w3.org/2000/svg}path':
                if j.attrib.get('class')!=None:
                    bond = j.attrib['class'].split('-')
                    if int(bond[-1]) in self.bonds_to_highlight and 'bond' in bond[0]:
                        j.attrib['style']=self.bond_diaplay.format(self.colors,self.colors)
                    elif int(bond[-1]) in self.atoms_to_highlight and 'atom' in bond[0]:
                        j.attrib['style']=self.atom_display.format(self.colors,self.colors)
            new_parsed[i]=j 
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        new_svg = ET.tostring(new_parsed,encoding='iso-8859-1',method='xml').decode('iso-8859-1')
        display(SVG(new_svg))

def MapAtomFromWeights(mol,weights,draw2d,contourLines=0,sigma=None,gridResolution=0.06): 
    # A modified version from rdkit SimilarityMaps.GetSimilarityMapFromWeights
    """
    Generates the similarity map for a molecule given the atomic weights.

    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    mol = rdMolDraw2D.PrepareMolForDrawing(mol,addChiralHs=False)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1)-mol.GetConformer().GetAtomPosition(idx2)).Length()
        else:
            sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0)-mol.GetConformer().GetAtomPosition(1)).Length()
        sigma = round(sigma, 2)
    sigmas = [sigma]*mol.GetNumAtoms()
    locs=[]
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x,p.y))
    draw2d.ClearDrawing()
    ps = Draw.ContourParams()
    ps.fillGrid=True
    ps.gridResolution=gridResolution
    ps.extraGridPadding = 1
    Draw.ContourAndDrawGaussians(draw2d,locs,weights,sigmas,nContours=contourLines,params=ps)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d

def get_numed_mol(smiles,addH = False,img_size=400):
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if addH:
        mol = Chem.AddHs(mol) 
    for i ,atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    drawer = Chem.Draw.MolDraw2DSVG(img_size, img_size)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol,addChiralHs=False)
    drawer.ClearDrawing()
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    return SVG(svg)

degrees = [0,1,2,3,4,5]
def atoms_by_order(rdkit_ix,max_atom):
    order_matrix = np.zeros((max_atom,max_atom),'float32')
    for i,j in enumerate(rdkit_ix):
        order_matrix[j,i] = 1
    return order_matrix
def connectivity_to_Matrix(array_rep, total_num_features,degree):#
    total_num = []
    mat = np.zeros((total_num_features, total_num_features),'float32') 
    if degree == 0:   
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[i,x] = 1        
        return mat
    else:
        for i in range(degree):
            atom_neighbors_list = array_rep[('atom_neighbors',i)].astype('int32')
            total_num.append(len(atom_neighbors_list))
        total_num = sum(total_num)
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[total_num + i,x] = 1
        return mat
def bond_features_by_degree(total_atoms,summed_degrees,degree):
    mat = np.zeros((total_atoms,10),'float32')
    total_num = []
    if degree == 0:
        for i,x in enumerate(summed_degrees[0]):
            mat[i] = x
        return mat
    else:
        for i in range(degree):
            total_num.append(len(summed_degrees[i]))
        total_num = sum(total_num)
        for i,x in enumerate(summed_degrees[degree]):
            mat[total_num + i] = x
        return mat
def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)
    
    Returns:
    ----------
    
    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]
    
    e.g.:
    
    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]
    
    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]  
    
    """
    bond_features_by_atom_by_degree = []
    for degree in degrees:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree #包含每个度数级别的原子的邻居化学键特征
if __name__ == "__main__":
    data_base = '/root/autodl-tmp/code2_stilltry/data'
    mol_token = Mol_Tokenizer(data_base + '/token_id.json')
   #token_ids,edges = mol_token.tokenize('CC1CCC2=CC(F)=CC3=C2N1C=C(C(O)=O)C3=O')
    motif_list,edge,ids = mol_token.tokenize('CC1CCC2=CC(F)=CC3=C2N1C=C(C(O)=O)C3=O')
    print(motif_list)
   # print(token_ids)
    q =[]
    for aa in motif_list:
        bb = mol_token.id_to_token.get(aa, None)
        q.append(bb)
    print(q)
    print(ids)
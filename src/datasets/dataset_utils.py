import torch
import pickle
from rdkit import Chem
import networkx as nx
import numpy as np
from torch_geometric.utils import sort_edge_index, to_dense_adj, remove_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from src.reduction.expansion import get_expanded_graph
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, calculate_base_properties, penalized_property
from src.reduction.reduction import get_comm_coarsed_graph, get_lap_coresed_graph, get_custom_coarsed_graph, get_expand_check
from src.reduction.expansion import get_expanded_graph
import src.utils as utils
from typing import Any, Sequence
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
import os.path as osp
from src.utils import to_dense
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors, Crippen
from src.analysis.rdkit_functions import qed
    
class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data
    
class SelectLogPTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data

def get_red_arg_dict(cfg):
    reduction_args = dict()
    reduction_args['reduction_type'] = cfg.dataset.reduction_type
    reduction_args['structual_dist'] = cfg.dataset.structual_dist
    reduction_args['filter'] = cfg.dataset.filter
    reduction_args['charged'] = cfg.dataset.charged

    if cfg.dataset.reduction_type == 'custom':
        reduction_args['split_ring'] = cfg.dataset.split_ring

    if cfg.dataset.reduction_type == 'comm':
        if cfg.dataset.max_condense_size is not None:
            max_condense_size = cfg.dataset.max_condense_size
            reduction_args['max_condense_size'] = max_condense_size
        else:
            reduction_args['max_condense_size'] = float('inf')

        if cfg.dataset.min_condense_size is not None:
            min_condense_size = cfg.dataset.min_condense_size
            reduction_args['min_condense_size'] = min_condense_size
        else:
            reduction_args['min_condense_size'] = 2
        reduction_args['resolution'] = cfg.dataset.resolution
        reduction_args['order'] = cfg.dataset.order
        reduction_args['reduce_frac'] = cfg.dataset.reduce_frac
    return reduction_args

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])
    
def filter_valid_mol(G, atom_decoder):
    if not nx.is_connected(G):
        print("Disconnected Graph")
        return None  # 如果图不连通，直接返回None
    
    try:
        data = from_networkx(G, group_node_attrs=['label'], group_edge_attrs=['bond_type'])
    except KeyError: return None
    data = Batch.from_data_list([data])
    dense_data, node_mask = utils.to_dense(data.x.squeeze(), data.edge_index, data.edge_attr.squeeze(), data.batch,
                                            x_classes=len(atom_decoder), e_classes=5, )
    dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E

    assert X.size(0) == 1
    atom_types = X[0].squeeze()
    edge_types = E[0].squeeze()
    mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
    smiles = mol2smiles(mol)
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(mol_frags) == 1:
                return G
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
    return None

def to_undirect(edge_index,edge_attrs):
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).contiguous()

    if isinstance(edge_attrs, list):
        _edge_attrs = []
        for edge_attr in edge_attrs:
            if len(edge_attr.shape) > 1:
                assert 'attr dim must be 1'
            else:
                edge_attr = torch.cat([edge_attr, edge_attr]).contiguous()
                _edge_attrs.append(edge_attr)
    
        full_edge_attrs = torch.stack(_edge_attrs,dim=-1)
        edge_index,full_edge_attrs  = sort_edge_index(edge_index,full_edge_attrs)
        edge_attrs = list(full_edge_attrs[:,i] for i in range(len(_edge_attrs)))
    else:
        edge_attrs = torch.cat([edge_attrs, edge_attrs]).contiguous()
        edge_index, edge_attrs = sort_edge_index(edge_index,edge_attrs)
    return sort_edge_index(edge_index,edge_attrs)

def charged_to_nx(data, atom_encoder, bonds, build_with_charges=True, cal_prop=True):
    if data is None: 
        return None
    elif isinstance(data,str):
        mol = Chem.MolFromSmiles(data) 
    else: mol = data
    
    G = nx.Graph() 
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()

        #if there are some atoms we may want to keep track of the formal charges
        #(either positive, negative, or both)
        if(build_with_charges):
            #gets the atom's formal charge
            atom_charge = atom.GetFormalCharge()

            #if the charge is not neutral
            if(atom_charge != 0):
                #this is necessary, as the sign "+" is lost when converting
                #atom_charge > 0 to a string. If charge < 0, the "-" is already embedded
                sign = ""
                if(atom_charge > 0):
                    sign = "+"

                #if the charge is not neutral, then its string in
                #the "types" dictionary is of the form <atom_symbol><formal charge>
                actual_atom_symbol = atom_symbol + sign + str(atom_charge)
                
                #check if the actual_atom_symbol is in the types dictionary.
                #if present, it means that we want to keep track of that
                #non-neutral version of the atom. Otherwise, we do not keep
                #the molecule.
                if(actual_atom_symbol in atom_encoder):
                    atom_symbol = actual_atom_symbol
                else:
                    return False
        
        G.add_node(atom_idx, label=atom_encoder[atom_symbol])

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bonds[bond.GetBondType()]
        G.add_edge(begin_idx, end_idx, bond_type=bond_type)
    if cal_prop:
        logP, qed, sa_score = calculate_base_properties(mol)
        P_logP = penalized_property(mol)
        G.graph['y'] = torch.tensor([[logP, qed, sa_score, P_logP]], dtype=torch.float32)
    return G

def nx_to_mol(G, atom_decoder, bonds):
    """
    将 NetworkX 图对象 G 转换为 RDKit 分子对象。
    
    参数:
    G (networkx.Graph): 表示分子结构的图对象，其中节点包含 'label' 表示原子类型，边包含 'bond_type' 表示键类型。
    atom_decoder (dict): 将数字标签转换回原子符号的字典。
    bond_decoder (dict): 将数字标签转换回键类型的字典。
    
    返回:
    Chem.Mol: RDKit 分子对象。
    """
    bond_decoder = list(bonds.keys())
    mol = Chem.RWMol()  # 使用可编辑的分子对象

    # 创建 RDKit 原子并添加到分子中
    idx_map = {}  # 将图节点映射到 RDKit 原子索引
    for node_idx, node_data in G.nodes(data=True):
        atom_label = node_data.get("label")
        atom_symbol = atom_decoder[atom_label]
        if atom_symbol is None:
            raise ValueError(f"未知的原子标签：{atom_label}")
        
        atom = Chem.Atom(atom_symbol)
        rd_atom_idx = mol.AddAtom(atom)
        idx_map[node_idx] = rd_atom_idx

    # 创建键并添加到分子中
    for start, end, edge_data in G.edges(data=True):
        bond_label = edge_data.get("bond_type")
        if bond_label == 0: continue
        bond_type = bond_decoder[bond_label]
        if bond_type is None:
            raise ValueError(f"未知的键类型标签：{bond_label}")
        
        mol.AddBond(idx_map[start], idx_map[end], bond_type)

    # 转换为 Chem.Mol 格式
    mol = mol.GetMol()
    Chem.Kekulize(mol)
    Chem.SanitizeMol(mol)  # 确保分子结构有效
    return mol

def init_single_coarsed_G(coarsed_G):
    x_count = torch.tensor(list(dict(coarsed_G.nodes.data('node_count')).values()), dtype=torch.long) - 1
    ring_num = torch.tensor(list(dict(coarsed_G.nodes.data('ring_num')).values()), dtype=torch.long)

    edges = list(coarsed_G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.shape[-1],2))
    edge_attr[:,0] = 0
    return x_count, ring_num, edge_index, edge_attr

def init_single_expanded_G(expanded_G):
    x_type_mask = torch.tensor(list(dict(expanded_G.nodes.data('type_mask',0)).values()), dtype=torch.long)
    edges = list(expanded_G.edges)
    masked_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # all available edge_index
    edge_type_mask = torch.tensor([item[-1] for item in list(expanded_G.edges.data('type_mask',0))], dtype=torch.long)
    return x_type_mask, masked_edge_index, edge_type_mask

def get_expanded_G_target(graph,expanded_G):
    traget_edge_label = {(u,v):{'edge_target':graph[u][v]['bond_type']} for (u,v) in graph.edges}
    nx.set_edge_attributes(expanded_G, traget_edge_label )

    x_target = torch.tensor(list(dict(graph.nodes.data('label',0)).values()), dtype=torch.long)
    edge_target = torch.tensor([item[-1] for item in list(expanded_G.edges.data('edge_target',0))], dtype=torch.long)
    return x_target, edge_target

def expand_graphs(coarsed_graphs, collate=True):
    data_list = []
    for graph in coarsed_graphs:
        reduced_G = nx.Graph()
        adjacency_matrix = graph[2]

        # 获取有效节点并创建节点属性
        node_count = graph[0][graph[0] != -1].numpy() + 1
        ring_nums = graph[1][graph[1] != -1].numpy()
        node_attrs = []
        nodes = []
        offset = 0
        for count, ring_num in zip(node_count,ring_nums):
            nodes.append(offset)
            node_attrs.append({'node_count': count, 'ring_num': ring_num, 'node_map':list(range(offset,offset+count))})
            offset += count
        nodes = np.array(nodes)
        edge_index = np.array(adjacency_matrix.nonzero())
        edge_index[:,0] = nodes[edge_index[:,0]]        # edge index map to the node labels
        edge_index[:,1] = nodes[edge_index[:,1]]

        reduced_G.add_nodes_from(zip(nodes, node_attrs))
        reduced_G.add_edges_from(edge_index)
        largest_cc = list(nx.connected_components(reduced_G))  # 获取最大连通子图的节点集合
        if len(largest_cc) > 1:
            reduced_G = reduced_G.subgraph(max(largest_cc,key=len)).copy()  # 创建最大连通子图的副本

        expanded_G = get_expanded_graph(reduced_G,reindex=True)
        x_type_mask, masked_edge_index, edge_type_mask = init_single_expanded_G(expanded_G)

        masked_edge_index, edge_type_mask = \
            to_undirect(masked_edge_index,edge_type_mask)
        data = Data(
            x=torch.empty_like(x_type_mask), 
            edge_index=torch.empty((2,0)), 
            edge_attr=torch.empty((1,0)), 

            x_type_mask=x_type_mask,
            masked_edge_index=masked_edge_index,
            edge_type_mask=edge_type_mask,
            )
        data_list.append(data)
    if collate:
        return Batch.from_data_list(data_list)
    else:
        return data_list

def repeat_batch(batch, sample_per_graph):
    data_list = batch.to_data_list()
    repeated_data_list = []
    for _ in range(sample_per_graph):
        repeated_data_list.extend([data.clone() for data in data_list])
    repeated_batch = Batch.from_data_list(repeated_data_list)
    return repeated_batch

def graph_from_scaffold(scaffold_smile: str, 
                        c_dataset_infos,
                        e_dataset_infos, 
                        max_condense_size:int=None,
                        min_condense_size:int=None,
                        reduce_frac:float=None,
                        ) -> Data:
    """
    Converts a SMILES string representing a molecular scaffold into a PyTorch Geometric Data object.
    :param scaffold_smile: SMILES string representing the molecular scaffold
    :return: PyG Data object
    """
    # Step 1: Initialize lists to store node and edge information
    G = charged_to_nx(scaffold_smile, e_dataset_infos.atom_encoder, e_dataset_infos.bonds)
    

    # Step 2: Construct the PyG Data object
    reduction_args = e_dataset_infos.reduction_args

    if reduction_args['reduction_type'] == 'comm':
        max_condense_size = max_condense_size if max_condense_size is not None else reduction_args['max_condense_size']
        min_condense_size = min_condense_size if min_condense_size is not None else reduction_args['min_condense_size']
        reduce_frac = reduce_frac if reduce_frac is not None else reduction_args['reduce_frac']
        coarsed_G = get_comm_coarsed_graph(
                                    G=G, 
                                    max_condense_size=max_condense_size, 
                                    min_condense_size=min_condense_size,
                                    reduce_frac=reduce_frac, 
                                    disturb_frac=0., 
                                    resolution=reduction_args['resolution'] * .5,
                                    order=reduction_args['order'],
                                    reindex=False, 
                                )
    elif reduction_args['reduction_type'] == 'custom':
        
        coarsed_G = get_custom_coarsed_graph(G=G,split_ring=reduction_args['split_ring'],reindex=False)
    else: raise NotImplementedError
    
    coarsed_G = get_expand_check(G, coarsed_G, e_dataset_infos.max_valencies)
    expanded_G = get_expanded_graph(coarsed_G, reindex=True)
    coarsed_G = nx.convert_node_labels_to_integers(coarsed_G)

    x_counts, ring_nums, edge_index, edge_attr = init_single_coarsed_G(coarsed_G)
    if coarsed_G.number_of_edges() != 0:
        edge_index, edge_attr = to_undirect(edge_index,edge_attr[:,1])

    x_type_mask, masked_edge_index, edge_type_mask = init_single_expanded_G(expanded_G)
    x_type_mask = x_type_mask + 1  # zero is encoded as null mask, so we just add one to it
    edge_type_mask = edge_type_mask + 1 
    x_target, edge_target = get_expanded_G_target(G, expanded_G)
    masked_edge_index, [edge_type_mask, edge_target] = to_undirect(masked_edge_index,[edge_type_mask, edge_target])

    allow_extend = torch.tensor(list(nx.get_node_attributes(coarsed_G, 'allow_extend').values()),dtype=bool)

    # Do not do onehot, the mask is applied on sampled_s
    coarsed_graph_data = Data(
                        x=x_counts,
                        x_aug=ring_nums, 
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        allow_extend=allow_extend,
                        )

    expanded_graph_data = Data(
                    x_target=x_target,
                    edge_target=edge_target,

                    x_type_mask=x_type_mask,  
                    masked_edge_index=masked_edge_index,
                    edge_type_mask=edge_type_mask,
                    )
    return coarsed_graph_data, expanded_graph_data

def fitler_to_nx(i, original_smile, atom_encoder, bonds, build_with_charges=True):
    smiles_2D = rstrip1(original_smile, "\n")

    #otteniamo la molecola dallo smiles_2D ed eseguiamo il preprocessing
    mol = clean_mol(smiles_2D, False)
    smiles_2D = Chem.MolToSmiles(mol)

    data = mol2graph(mol, atom_encoder, bonds, i, smiles_2D, True, True)

    if(data == None):
        return None
    atom_decoder = list(atom_encoder.keys())
    reconstructed_mol        = graph2mol(data, atom_decoder)
    try:
        reconstructed_mol    = clean_mol(reconstructed_mol, False)
        reconstructed_smiles = Chem.MolToSmiles(reconstructed_mol)
    except:
        return None

    if(smiles_2D == reconstructed_smiles):
        tmp_mol_original_smile       = mol
        tmp_mol_smiles_2D            = reconstructed_mol
        print_smiles                 = False

        tmp_mol_original_smile_qed   = qed(tmp_mol_original_smile)
        tmp_mol_original_smile_logp  = Crippen.MolLogP(tmp_mol_original_smile)
        tmp_mol_original_smile_mw    = rdMolDescriptors.CalcExactMolWt(tmp_mol_original_smile)

        tmp_mol_smiles_2D_qed        = qed(tmp_mol_smiles_2D)
        tmp_mol_smiles_2D_logp       = Crippen.MolLogP(tmp_mol_smiles_2D)
        tmp_mol_smiles_2D_mw         = rdMolDescriptors.CalcExactMolWt(tmp_mol_smiles_2D)

        if(abs(tmp_mol_original_smile_qed - tmp_mol_smiles_2D_qed) > 1e-5):
            print_smiles=True
        if(abs(tmp_mol_original_smile_logp - tmp_mol_smiles_2D_logp) > 1e-5):
            print_smiles=True
        if(abs(tmp_mol_original_smile_mw - tmp_mol_smiles_2D_mw) > 4):
            print_smiles=True
        
        if(print_smiles):
            print(f"Smiles [{reconstructed_smiles}] reconstruct property fails")
            reconstructed_smiles = None

    if reconstructed_smiles is not None and smiles_2D == reconstructed_smiles: 
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(reconstructed_mol, asMols=True, sanitizeFrags=True)
            if len(mol_frags) == 1:
                return charged_to_nx(reconstructed_smiles, atom_encoder, bonds, build_with_charges=build_with_charges)

        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")

class coarsed_sample_loader:
    def __init__(self, 
                 num_batch, 
                 num_nodes:int=None, 
                 coarsed_scaffold:str=None):
        self.num_batch = num_batch
        self.num_nodes = num_nodes
        self.scaffold_X = None
        self.scaffold_X_aug = None
        self.scaffold_E = None
        self.allow_extend = None

        if coarsed_scaffold is not None:
            self.scaffold_X = coarsed_scaffold.x.long()
            self.scaffold_X_aug = coarsed_scaffold.x_aug.long()
            self.allow_extend = coarsed_scaffold.allow_extend
            if coarsed_scaffold.edge_index.numel()==0:
                self.scaffold_E = torch.zeros((1,1),dtype=torch.long)
            else:
                self.scaffold_E = to_dense_adj(
                    edge_index=coarsed_scaffold.edge_index.long(),
                    edge_attr=coarsed_scaffold.edge_attr.long())
                
    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for i in range(self.num_batch):
            yield {
                'num_nodes': self.num_nodes,
                'scaffold_X': self.scaffold_X,
                'scaffold_X_aug': self.scaffold_X_aug,
                'scaffold_E': self.scaffold_E,
                'allow_extend': self.allow_extend,
            }

class refinement_sample_loader:
    def __init__(self, 
                 num_batch, 
                 batch_size, 
                 coarsed_graphs, 
                 expanded_scaffold:str=None,
                 ):
        self.num_batch = num_batch
        self.batch_size = batch_size

        expanded_batch = expand_graphs(coarsed_graphs)
        self.data_loader = iter(DataLoader(expanded_batch, batch_size=batch_size))
        
        self.scaffold_X = None
        self.scaffold_E = None
        self.scaffold_X_type_mask = None
        self.scaffold_E_type_mask = None

        if expanded_scaffold:
            self.scaffold_X = expanded_scaffold.x_target
            self.scaffold_E = to_dense_adj(
                edge_index=expanded_scaffold.masked_edge_index,
                edge_attr=expanded_scaffold.edge_target)
            
            self.scaffold_X_type_mask = expanded_scaffold.x_type_mask
            self.scaffold_E_type_mask = to_dense_adj(
                edge_index=expanded_scaffold.masked_edge_index,
                edge_attr=expanded_scaffold.edge_type_mask)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for i in range(self.num_batch):
            yield {
                'expanded_batch': next(self.data_loader),
                'scaffold_X': self.scaffold_X,
                'scaffold_E': self.scaffold_E,
                'scaffold_X_type_mask': self.scaffold_E,
                'scaffold_E_type_mask': self.scaffold_E,
            }

class GraphInMemoryDataset(Data):
    def __init__(self, x: OptTensor = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None,
                 original_smiles = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.original_smiles = original_smiles

###############################################################################
# FreeGress stuff
def rstrip1(s, c):
    return s[:-1] if s[-1]==c else s

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors, Crippen
from src.analysis.rdkit_functions import build_molecule, qed
from torch_geometric.data import Batch
import torch.nn.functional as F



def clean_mol(mol, uncharge=True):
    if(isinstance(mol, str)):
        mol = Chem.MolFromSmiles(mol)

    Chem.RemoveStereochemistry(mol)
    if(uncharge):
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    
    return mol

def graph2mol(data, atom_decoder):
    data = Batch.from_data_list([data])

    dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    dense_data = dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E

    assert X.size(0) == 1
    atom_types = X[0]
    edge_types = E[0]

    reconstructed_mol = build_molecule(atom_types, edge_types, atom_decoder)
    return reconstructed_mol

def mol2graph(mol, types, bonds, i, original_smiles = None, estimate_guidance = True, build_with_charges = False):
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()

        #if there are some atoms we may want to keep track of the formal charges
        #(either positive, negative, or both)
        if(build_with_charges):
            #gets the atom's formal charge
            atom_charge = atom.GetFormalCharge()

            #if the charge is not neutral
            if(atom_charge != 0):
                #this is necessary, as the sign "+" is lost when converting
                #atom_charge > 0 to a string. If charge < 0, the "-" is already embedded
                sign = ""
                if(atom_charge > 0):
                    sign = "+"

                #if the charge is not neutral, then its string in
                #the "types" dictionary is of the form <atom_symbol><formal charge>
                actual_atom_symbol = atom_symbol + sign + str(atom_charge)
                
                #check if the actual_atom_symbol is in the types dictionary.
                #if present, it means that we want to keep track of that
                #non-neutral version of the atom. Otherwise, we do not keep
                #the molecule.
                if(actual_atom_symbol in types):
                    atom_symbol = actual_atom_symbol
                else:
                    print(actual_atom_symbol)
                    return None
        
        type_idx.append(types[atom_symbol])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    if len(row) == 0:
        print("Number of rows = 0")
        return None

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    return GraphInMemoryDataset(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                                y=y, idx=i, original_smiles = original_smiles)
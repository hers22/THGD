import os
import torch
import pathlib
import networkx as nx
import torch_geometric.utils
from tqdm.auto import tqdm

from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
import torch_geometric
import torch.nn.functional as F
import torch_scatter

from src import utils
from src.reduction.reduction import get_comm_coarsed_graph, get_custom_coarsed_graph
from src.reduction.expansion import get_expanded_graph
from src.diffusion.distributions import DistributionNodes
from src.datasets.dataset_utils import init_single_coarsed_G, init_single_expanded_G, get_expanded_G_target

from joblib import Parallel, delayed
from src.datasets.dataset_utils import to_undirect
from rdkit import Chem
from src.reduction.structural_types import type_mask
from src.datasets.dataset_utils import get_red_arg_dict
from src.datasets.dataset_utils import (fitler_to_nx, charged_to_nx)


class AbstractMolecularDataset(InMemoryDataset):
    charged_atoms = ['N+1', 'N-1', 'O-1', 'S-1']
    num_type_mask = len(type_mask)
    ring_size = 0
    def __init__(self, stage, root, graph_type, 
                 transform=None, pre_transform=None, pre_filter=None,
                 reduction_args:dict = {}, extra_charged_atoms=[]):
        self.stage = stage
        self.n_jobs = -1
        self.structual_dist = reduction_args['structual_dist'] 
        self.reduction_args = reduction_args

        if 'max_condense_size' in reduction_args:
            self.max_condense_size = reduction_args['max_condense_size'] 
        if self.reduction_args['charged'] and self.stage == 'train':
            charged_atoms = self.charged_atoms + extra_charged_atoms
            print(f"Adding charged atoms: {charged_atoms}")
            self.atom_decoder += charged_atoms
            self.atom_encoder = {atom:i for i,atom in enumerate(self.atom_decoder)}
            
        self.arg_string = '_'.join([f'[{key}:{val}]' for (key,val) in self.reduction_args.items()])
        

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root=root,transform=transform, pre_filter=pre_filter, pre_transform=pre_transform)
        if graph_type == 'expand':
            self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.processed_paths[3 + self.file_idx]))
        else:
            self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.processed_paths[self.file_idx]))

    @property
    def raw_file_names(self): raise NotImplementedError
    
    def download(self): raise NotImplementedError
    
    def get_smiles_list(self): raise NotImplementedError
    
    def get_nx_graphs(self): 
        smile_list = self.get_smiles_list()

        if self.reduction_args['filter']:
            print(f"Size before filter: {len(smile_list)}")
            print("Converting smiles...")

            graphs = Parallel(n_jobs=-1, batch_size='auto')(
                delayed(fitler_to_nx)(i, s, self.atom_encoder, self.bonds, build_with_charges=self.reduction_args['charged']) for i, s in enumerate(tqdm(smile_list))
            )
        else:
            graphs = Parallel(n_jobs=-1, batch_size='auto')(
                delayed(charged_to_nx)(s, self.atom_encoder, self.bonds, build_with_charges=self.reduction_args['charged']) for i, s in enumerate(tqdm(smile_list))
            )
        graphs = [g for g in graphs if (g is not None)]
        print(f"Size after converted: {len(graphs)}, loss rate: {100*(len(smile_list)-len(graphs))/len(smile_list)}%")
        return graphs
    
    @property
    def processed_file_names(self):
        return [
                f'coarsed_train_{self.arg_string}.pt',
                f'coarsed_val_{self.arg_string}.pt',
                f'coarsed_test_{self.arg_string}.pt',
                f'expanded_train_{self.arg_string}.pt',
                f'expanded_val_{self.arg_string}.pt',
                f'expanded_test_{self.arg_string}.pt',
                f'coarsed_distribution_{self.arg_string}.pt',
                f'expanded_distribution_{self.arg_string}.pt'
                ]
    
    def process_single_graph(self, G):
        if self.reduction_args['reduction_type'] == 'comm':
            reduced_G = get_comm_coarsed_graph(
                                        G=G, 
                                        max_condense_size=self.max_condense_size, 
                                        min_condense_size=self.reduction_args['min_condense_size'], 
                                        reduce_frac=self.reduction_args['reduce_frac'], 
                                        resolution=self.reduction_args['resolution'],
                                        reindex=False, 
                                    )
        elif self.reduction_args['reduction_type'] == 'custom':
            reduced_G = get_custom_coarsed_graph(
                                        G=G, 
                                        split_ring=self.reduction_args['split_ring'],
                                        reindex=False, 
                                    )
        expanded_G = get_expanded_graph(reduced_G)
        reduced_G = nx.convert_node_labels_to_integers(reduced_G)

        max_ring = max(nx.get_node_attributes(reduced_G, 'ring_num').values())
        max_size = max(nx.get_node_attributes(reduced_G, 'node_count').values())
        # if max_size > self.max_condense_size:
        #     print(f"Found a graph with max node count bigger than {self.max_node_count}. Skip.")
        #     return None, None, 0, 0
        return (reduced_G, expanded_G, max_ring, max_size)


    def process(self):
        print('Process args:', self.arg_string)

        graphs = self.get_nx_graphs()
        print("Reducing & Expanding Graph...") 

        if self.file_idx!=0:
            _, node_distribution, ring_num_distribution, _ = torch.load(os.path.join(self.processed_dir, self.processed_file_names[-2]))
            self.ring_size = len(ring_num_distribution)
            self.max_condense_size = node_distribution.shape[-1]

        reduced_expanded_graphs = Parallel(n_jobs=self.n_jobs,  batch_size='auto')(
            delayed(self.process_single_graph)(g) for g in tqdm(graphs)
        )

        reduced_graphs, expanded_graphs, ring_sizes, max_condense_size = zip(*reduced_expanded_graphs)
        if self.file_idx==0:
            self.ring_size = max(ring_sizes) + 1
            self.max_condense_size = max(max_condense_size)
            
        self.num_type_mask = self.num_type_mask + self.ring_size - 1
        
        print("Initializing Expanded Graph...")
        data_list = self.init_expanded_graphs(expanded_graphs, graphs,cal_dist=True if self.file_idx==0 else False)
        
        if self.file_idx==0:
            torch.save((self.node_mask_distribution, self.edge_mask_distribution), \
                       os.path.join(self.processed_dir, self.processed_file_names[-1]))
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[3 + self.file_idx]))

        print("Initializing Coarsed Graph...")
        data_list = self.init_coarsed_graphs(reduced_graphs, cal_dist=True if self.file_idx==0 else False)
        
        if self.file_idx==0:
            torch.save((self.graph_size_distribution, self.node_count_distribution, self.ring_num_distribution, self.edge_count_distribution), \
                       os.path.join(self.processed_dir, self.processed_file_names[-2]))
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        if self.pre_filter is not None:
            data_list = [self.pre_filter(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[self.file_idx]))

    def get_masked_distribution(self, masks, attrs, num_attrs):
        '''
            return 
                Tensor: mask_types x num_attrs 
        '''
        distribution = torch.zeros((self.num_type_mask),num_attrs,dtype=torch.float32)
        for mask_type in range(self.num_type_mask):
            type_attr = attrs[masks == mask_type]
            distribution[mask_type] = torch_scatter.scatter_add(src=torch.ones(len(type_attr)),index=type_attr,dim_size=num_attrs)
        return distribution

    def init_coarsed_graphs(self, coarsed_graphs, cal_dist=False):
        data_list = []
        if cal_dist:
            self.graph_size_distribution = torch.zeros(max(g.number_of_nodes() for g in coarsed_graphs if g is not None) + 2,dtype=torch.float32)
            self.node_count_distribution = torch.zeros(self.max_condense_size,dtype=torch.float32)
            self.ring_num_distribution = torch.zeros(self.ring_size,dtype=torch.float32) 
            self.edge_count_distribution = torch.zeros(2,dtype=torch.float32)
        for i, coarsed_G in enumerate(tqdm(coarsed_graphs)):
            if coarsed_G is None or coarsed_G.number_of_edges() == 0: continue
            num_nodes = coarsed_G.number_of_nodes()

            x_count, ring_num, edge_index, edge_attr = init_single_coarsed_G(coarsed_G)  
            edge_index, edge_attr = to_undirect(edge_index,edge_attr)

            if cal_dist:
                self.graph_size_distribution[num_nodes] += 1 
                self.node_count_distribution.scatter_add_(0, x_count, torch.ones(x_count.shape[0]))
                self.ring_num_distribution.scatter_add_(0, ring_num, torch.ones(ring_num.shape[0]))
                self.edge_count_distribution[1] += edge_index.shape[1]
                self.edge_count_distribution[0] += x_count.shape[0] ** 2 - edge_index.shape[1]

            if max(x_count) >= self.max_condense_size: continue
            data = Data(x=F.one_hot(x_count.view(-1), self.max_condense_size).float().view(-1, self.max_condense_size), 
                        x_aug=F.one_hot(ring_num.view(-1), self.ring_size).float().view(-1, self.ring_size), 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=torch.tensor(coarsed_G.graph['y']) if 'y' in coarsed_G.graph else torch.empty((1,0)), 
                        dtype=torch.float)
            data_list.append(data)

        if cal_dist:
            self.graph_size_distribution = self.graph_size_distribution / self.graph_size_distribution.sum()
            self.node_count_distribution = self.node_count_distribution / self.node_count_distribution.sum()
            self.ring_num_distribution = self.ring_num_distribution / self.ring_num_distribution.sum()
            self.edge_count_distribution = self.edge_count_distribution / self.edge_count_distribution.sum()
        return data_list

    def init_expanded_graphs(self, expanded_graphs, graphs, cal_dist=False):
        data_list = []
        num_node_attr = len(self.atom_decoder)
        num_bond_attr = len(self.bonds)
        if cal_dist:
            self.node_mask_distribution = torch.zeros((self.num_type_mask,num_node_attr),dtype=torch.float32)
            self.edge_mask_distribution = torch.zeros((self.num_type_mask,num_bond_attr),dtype=torch.float32)
        for i, (expanded_G,graph) in enumerate(tqdm(zip(expanded_graphs,graphs),total=len(graphs))):
            if expanded_G is None: continue
            x_target, edge_target = get_expanded_G_target(graph,expanded_G)

            x_type_mask, masked_edge_index, edge_type_mask = init_single_expanded_G(expanded_G)

            if cal_dist:
                if self.structual_dist:
                    self.node_mask_distribution = self.node_mask_distribution + self.get_masked_distribution(x_type_mask, x_target, num_node_attr)
                    self.edge_mask_distribution = self.edge_mask_distribution + self.get_masked_distribution(edge_type_mask, edge_target, num_bond_attr)

                else:
                    node_mask_distribution = torch_scatter.scatter_add(src=torch.ones(len(x_target)),index=x_target,dim_size=num_node_attr)
                    self.node_mask_distribution = self.node_mask_distribution + node_mask_distribution.unsqueeze(0)

                    edge_mask_distribution = torch_scatter.scatter_add(src=torch.ones(len(edge_target)),index=edge_target,dim_size=num_bond_attr)
                    self.edge_mask_distribution = self.edge_mask_distribution + edge_mask_distribution.unsqueeze(0)

            masked_edge_index, [edge_type_mask,edge_target] = \
                to_undirect(masked_edge_index,[edge_type_mask,edge_target])
            
            data = Data(x=torch.empty_like(x_type_mask), 
                        edge_index=torch.empty((2,0)), 
                        edge_attr=torch.empty((1,0)), 

                        x_type_mask=x_type_mask,
                        masked_edge_index=masked_edge_index,
                        edge_type_mask=edge_type_mask,

                        x_target=x_target,
                        edge_target=edge_target,

                        y =  torch.tensor(expanded_G.graph['y']) if 'y' in expanded_G.graph else torch.empty((1,0)),
                    )
            data_list.append(data)
        if cal_dist:
            self.node_mask_distribution = self.node_mask_distribution
            denorminator = self.node_mask_distribution.sum(-1,keepdim=True)
            denorminator[denorminator==0] = 1e-6
            self.node_mask_distribution = self.node_mask_distribution / denorminator
            
            self.edge_mask_distribution = self.edge_mask_distribution
            denorminator = self.edge_mask_distribution.sum(-1,keepdim=True)
            denorminator[denorminator==0] = 1e-6
            self.edge_mask_distribution = self.edge_mask_distribution / denorminator
        return data_list

class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.reduction_args = get_red_arg_dict(cfg)
        self.arg_string = '_'.join([f'[{key}:{val}]' for (key,val) in self.reduction_args.items()])
        

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir,'processed')
        dist_file = os.path.join(root_path,f'coarsed_distribution_{self.arg_string}.pt')                             
        self.n_nodes, self.node_types, self.agument_node_types, self.edge_types \
            = torch.load(dist_file,weights_only=False)
        
        self.max_condense_size = self.n_nodes.shape[-1]
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        print('example data:')
        print(example_batch)
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1) + example_batch['x_aug'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'guidance': example_batch['y'].size(1),
                           'y': 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'X_aug': example_batch['x_aug'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}

class AbstractExpandDatasetInfos:
    charged_valencies = [2, 4, 3, 3]
    charged_max_valencies = [3, 4, 4, 6]
    charged_weights = [14,14,16,32.065]
    def __init__(self, cfg, extra_charged_valencies=[], extra_charged_max_valencies=[], extra_charged_weights=[]):
        self.name = None
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.reduction_args = get_red_arg_dict(cfg)
        self.arg_string = '_'.join([f'[{key}:{val}]' for (key,val) in self.reduction_args.items()])

        # add charged atoms
        if cfg.dataset.charged:
            self.valencies += self.charged_valencies+extra_charged_valencies
            self.max_valencies += self.charged_max_valencies+extra_charged_max_valencies
            self.charged_weights += extra_charged_weights
            for i in range(len(self.charged_weights)):
                self.atom_weights[len(self.atom_weights)+i+1] = self.charged_weights[i]
            
        
        self.num_atom_types = len(self.atom_decoder)
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir,'processed')
        dist_file = os.path.join(root_path,f'expanded_distribution_{self.arg_string}.pt')                             

        self.node_types,self.edge_types = torch.load(dist_file,weights_only=False)
        self.num_mask_type = self.node_types.shape[0] + 1

        self.valency_distribution = None

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        print('example data:')
        print(example_batch)
        x_type_mask=example_batch.x_type_mask + 1 # encode null mask
        masked_edge_index=example_batch.masked_edge_index 
        edge_type_mask=example_batch.edge_type_mask + 1 # encode null mask
        batch = example_batch.batch
        
        x_target=example_batch.x_target
        edge_target=example_batch.edge_target
        edge_exist_mask = edge_target!=0
        edge_target = edge_target[edge_exist_mask]
        edge_index = masked_edge_index[:,edge_exist_mask]

        # has_edge=has_edge,

        X_type_mask, E_type_mask = utils.masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, batch)
        ex_dense, node_mask = utils.to_dense(x_target, edge_index, edge_target, batch, x_classes=self.node_types.shape[-1], e_classes=self.edge_types.shape[-1])

        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask,
                        'X_type_mask':X_type_mask,'E_type_mask':E_type_mask,
        }

        self.input_dims = {'X': self.node_types.shape[-1] + self.num_mask_type,
                           'E': self.edge_types.shape[-1] + self.num_mask_type,
                           'guidance': example_batch['y'].size(1),
                           'y': 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.shape[-1]
        self.input_dims['E'] += ex_extra_feat.E.shape[-1]
        self.input_dims['y'] += ex_extra_feat.y.shape[-1]

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.shape[-1]
        self.input_dims['E'] += ex_extra_molecular_feat.E.shape[-1]
        self.input_dims['y'] += ex_extra_molecular_feat.y.shape[-1]
        self.output_dims = {'X': self.node_types.shape[-1],
                            'E': self.edge_types.shape[-1],
                            'y': 0}

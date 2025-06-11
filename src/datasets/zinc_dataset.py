
import os
import pathlib
import torch
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from src.datasets.abstract_dataset import get_red_arg_dict, AbstractMolecularDataset, MolecularDataModule, AbstractDatasetInfos, AbstractExpandDatasetInfos
from src.datasets.dataset_utils import (fitler_to_nx, 
                                        charged_to_nx,
                                        RemoveYTransform,
                                        filter_valid_mol,
                                        SelectPenalizedLogPTransform)
from rdkit.Chem.rdchem import BondType as BT



atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B':4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9}
atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S']
bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}


class ZINC250kDataset(AbstractMolecularDataset):
    atom_encoder = atom_encoder
    atom_decoder = atom_decoder
    bonds = bonds

    def __init__(self, stage, root, graph_type='expand',
                 transform=None, pre_transform=None, reduction_args={}):
        super().__init__(stage=stage, root=root, graph_type=graph_type, 
                         transform=transform, pre_transform=pre_transform, 
                         reduction_args=reduction_args)

    def download(self): pass

    @property
    def raw_file_names(self):
        return ['train.txt', 'valid.txt', 
                'test-800.txt',
                # 'test.txt',
                ]
    
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

    def filtered_to_nx(self, smile):
        G = charged_to_nx(smile, self.atom_encoder, self.bonds)
        return filter_valid_mol(G, self.atom_decoder)

    def get_smiles_list(self):
        smile_list = open(os.path.join(self.raw_dir,self.raw_file_names[self.file_idx])).readlines()
        return smile_list

class ZINC250kDataModule(MolecularDataModule):
    def __init__(self, cfg, guidance=False, target='PlogP'):
        self.datadir = cfg.dataset.datadir
        self.graph_type = cfg.dataset.graph_type

        self.reduction_args = get_red_arg_dict(cfg)

        transform = RemoveYTransform()
        if guidance:
            if target == 'PlogP':
                transform = SelectPenalizedLogPTransform()
            else:
                transform = None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            'train': ZINC250kDataset(stage='train', root=root_path,
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'val': ZINC250kDataset(stage='val', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'test': ZINC250kDataset(stage='test', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args)
            }
        super().__init__(cfg, datasets)


class CoarseZINC250kInfos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.name = 'coarsed_zinc250k'
        super().__init__(cfg)


class ExpandZINC250kInfos(AbstractExpandDatasetInfos):
    def __init__(self, cfg):

        self.name = 'expanded_zinc250k'
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder

        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2]
        self.max_valencies = [4, 3, 2, 1, 3, 1, 1, 3, 5, 6]
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 32.065, 11: 14,12: 14, 13: 16,14: 32.065}

        self.bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        self.max_weight = 3000

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                     1.4545e-05, 2.4241e-05, 7.2723e-05, 3.0059e-04, 7.5632e-04, 2.9816e-03,
                                     4.8821e-03, 7.2965e-03, 1.1524e-02, 1.7788e-02, 2.5860e-02, 3.5537e-02,
                                     4.7876e-02, 5.9385e-02, 7.2107e-02, 8.2778e-02, 7.5113e-02, 8.4087e-02,
                                     9.1970e-02, 9.0656e-02, 7.6320e-02, 6.1994e-02, 3.9241e-02, 3.0810e-02,
                                     2.3979e-02, 1.8874e-02, 1.4637e-02, 1.0162e-02, 6.9377e-03, 3.9707e-03,
                                     1.5466e-03, 5.1391e-04, 4.8482e-06])
                                        
        super().__init__(cfg)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0000, 0.1156, 0.2933, 0.3520, 0.2311, 0.0032, 0.0048])


if __name__ == '__main__':
    import yaml
    from omegaconf import DictConfig
    with open('/data2/chensm22/HRS/MHdiff/configs/zinc250k_exp/c_split.yaml', 'r') as file:
        cfg = {'dataset':yaml.safe_load(file),
               'train':{'batch_size':32, 'num_workers': 4},
               'general':{'name':'test'}
               }
    
    ZINC250kDataModule(cfg=DictConfig(cfg))
from rdkit.Chem.rdchem import BondType as BT

import os
import pathlib
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from src.datasets.abstract_dataset import (
    AbstractDatasetInfos, 
    AbstractMolecularDataset, 
    MolecularDataModule, 
    AbstractExpandDatasetInfos,
    get_red_arg_dict)
from src.datasets.dataset_utils import (
                                        RemoveYTransform,
                                        filter_valid_mol,
                                        SelectLogPTransform)

atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5,
                'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}


class GuacamolDataset(AbstractMolecularDataset):
    atom_decoder = atom_decoder
    atom_encoder = atom_encoder
    bonds = bonds

    def __init__(self, stage, root, graph_type='expand',
                 transform=None, pre_transform=None, pre_filter=None,
                 reduction_args={}):

        super().__init__(stage=stage, root=root, graph_type=graph_type, 
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                         reduction_args=reduction_args, extra_charged_atoms=['S+1'])

    @property
    def raw_file_names(self):
        return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    def download(self): pass

    def get_smiles_list(self): 
        smile_list = open(os.path.join(self.raw_dir,self.raw_file_names[self.file_idx])).readlines()
        smile_list = smile_list[:]
        return smile_list

class GuacamolDataModule(MolecularDataModule):
    def __init__(self, cfg, guidance=False):
        self.datadir = cfg.dataset.datadir
        self.graph_type = cfg.dataset.graph_type

        self.reduction_args = get_red_arg_dict(cfg)

        transform = RemoveYTransform() if not guidance else None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            'train': GuacamolDataset(stage='train', root=root_path,
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'val': GuacamolDataset(stage='val', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'test': GuacamolDataset(stage='test', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args)}
        super().__init__(cfg, datasets)


class CoarseGuacamolInfos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.name = 'coarsed_Guacamol'
        super().__init__(cfg)


class ExpandGuacamolInfos(AbstractExpandDatasetInfos):
    def __init__(self, cfg):
        self.name = 'expanded_Guacamol'

        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.max_weight = 1000
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]
        self.max_valencies = [4, 3, 2, 1, 3, 1, 1, 5, 5, 6, 2, 4] # need to fill this if you want to contrain scaffold
        self.bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 78.97, 12: 28.09}
        self.n_nodes = torch.tensor([0, 0, 3.5760e-06, 2.7893e-05, 6.9374e-05, 1.6020e-04,
                                     2.8036e-04, 4.3484e-04, 7.3022e-04, 1.1722e-03, 1.7830e-03, 2.8129e-03,
                                     4.0981e-03, 5.5421e-03, 7.9645e-03, 1.0824e-02, 1.4459e-02, 1.8818e-02,
                                     2.3961e-02, 2.9558e-02, 3.6324e-02, 4.1931e-02, 4.8105e-02, 5.2316e-02,
                                     5.6601e-02, 5.7483e-02, 5.6685e-02, 5.2317e-02, 5.2107e-02, 4.9651e-02,
                                     4.8100e-02, 4.4363e-02, 4.0704e-02, 3.5719e-02, 3.1685e-02, 2.6821e-02,
                                     2.2542e-02, 1.8591e-02, 1.6114e-02, 1.3399e-02, 1.1543e-02, 9.6116e-03,
                                     8.4744e-03, 6.9532e-03, 6.2001e-03, 4.9921e-03, 4.4378e-03, 3.5803e-03,
                                     3.3078e-03, 2.7085e-03, 2.6784e-03, 2.2050e-03, 2.0533e-03, 1.5598e-03,
                                     1.5177e-03, 9.8626e-04, 8.6396e-04, 5.6429e-04, 5.0422e-04, 2.9323e-04,
                                     2.2243e-04, 9.8697e-05, 9.9413e-05, 6.0077e-05, 6.9374e-05, 3.0754e-05,
                                     3.5045e-05, 1.6450e-05, 2.1456e-05, 1.2874e-05, 1.2158e-05, 5.7216e-06,
                                     7.1520e-06, 2.8608e-06, 2.8608e-06, 7.1520e-07, 2.8608e-06, 1.4304e-06,
                                     7.1520e-07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.1520e-07, 0.0000e+00,
                                     1.4304e-06, 7.1520e-07, 7.1520e-07, 0.0000e+00, 1.4304e-06])

        charged_valencies = [1]
        charged_max_valencies = [5]
        charged_weights = [32.065]
        super().__init__(cfg, charged_valencies, charged_max_valencies, charged_weights)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0: 7] = torch.tensor([0.0000, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053])


if __name__ == '__main__':
    import yaml
    from omegaconf import DictConfig
    with open('/data2/chensm22/HRS/MHdiff/configs/guacamol_exp/e_custom.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    GuacamolDataModule(cfg=DictConfig(cfg))
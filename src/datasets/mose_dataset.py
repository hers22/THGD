from rdkit.Chem.rdchem import BondType as BT

import os
import pathlib
import torch
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from src.datasets.abstract_dataset import get_red_arg_dict, AbstractMolecularDataset, MolecularDataModule, AbstractExpandDatasetInfos, AbstractDatasetInfos
from src.datasets.dataset_utils import (
                                        fitler_to_nx,
                                        RemoveYTransform)


atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}


class MoseDataset(AbstractMolecularDataset):
    atom_decoder = atom_decoder
    atom_encoder = atom_encoder
    bonds = bonds

    def __init__(self, stage, root, graph_type='expand',
                 transform=None, pre_transform=None, reduction_args={}):
        self.reduction_args = reduction_args
        super().__init__(stage=stage, root=root, graph_type=graph_type, 
                         transform=transform, pre_transform=pre_transform, 
                         reduction_args=reduction_args)
        

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']
    
    def download(self): pass

    def get_smiles_list(self):
        df = pd.read_csv(os.path.join(self.raw_dir,self.raw_file_names[self.file_idx]))
        smiles_list = df.loc[:,'SMILES']
        return smiles_list
        

class MoseDataModule(MolecularDataModule):
    def __init__(self, cfg, guidance=False):
        self.datadir = cfg.dataset.datadir
        self.graph_type = cfg.dataset.graph_type

        self.reduction_args = get_red_arg_dict(cfg)

        transform = RemoveYTransform() if not guidance else None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            'train': MoseDataset(stage='train', root=root_path,
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'val': MoseDataset(stage='val', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args),
            'test': MoseDataset(stage='test', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    reduction_args=self.reduction_args)}
        super().__init__(cfg, datasets)


class CoarseMoseInfos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.name = 'coarsed_mose'
        super().__init__(cfg)


class ExpandMoseInfos(AbstractExpandDatasetInfos):
    def __init__(self, cfg):
        self.name = 'expanded_mose'
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.valencies = [4, 3, 4, 2, 1, 1, 1]
        self.max_valencies = [4, 3, 6, 2, 1, 1, 1]
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9}
        self.bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        self.max_weight = 350

        self.n_nodes = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.097634362347889692e-06,
                                     1.858580617408733815e-05, 5.007842264603823423e-05, 5.678996240021660924e-05,
                                     1.244216400664299726e-04, 4.486406978685408831e-04, 2.253012731671333313e-03,
                                     3.231865121051669121e-03, 6.709992419928312302e-03, 2.289564721286296844e-02,
                                     5.411050841212272644e-02, 1.099515631794929504e-01, 1.223291903734207153e-01,
                                     1.280680745840072632e-01, 1.445975750684738159e-01, 1.505961418151855469e-01,
                                     1.436946094036102295e-01, 9.265746921300888062e-02, 1.820066757500171661e-02,
                                     2.065089574898593128e-06])
        

        super().__init__(cfg)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485])


if __name__ == '__main__':
    import yaml
    from omegaconf import DictConfig
    with open('/data2/chensm22/HRS/MHdiff/configs/mose_exp/e_custom.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    MoseDataModule(cfg=DictConfig(cfg))
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
from src.datasets.dataset_utils import (to_nx, 
                                        RemoveYTransform,
                                        filter_valid_mol,
                                        SelectLogPTransform)
atom_encoder = {'C': 0, 'N': 1, 'O':2,  'F': 3, 'P': 4, 'S':5, 'Si': 6}
atom_decoder = ['C', 'N', 'O', 'F', 'P', 'S', 'Si']
bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}


class PolymerDataset(AbstractMolecularDataset):
    atom_decoder = atom_decoder
    atom_encoder = atom_encoder
    bonds = bonds

    def __init__(self, stage, root, remove_h=None, graph_type='expand',
                 transform=None, pre_transform=None, pre_filter=None,
                 reduction_args={}):

        super().__init__(stage=stage, root=root, graph_type=graph_type, 
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                         reduction_args=reduction_args)

    @property
    def raw_file_names(self):
        return ['train.txt', 'valid.txt', 'test.txt']

    def download(self): pass

    def filtered_to_nx(self, smile):
        G = to_nx(smile, self.atom_encoder, self.bonds)
        return filter_valid_mol(G, self.atom_decoder)

    def get_nx_graphs(self): 
        smile_list = open(os.path.join(self.raw_dir,self.raw_file_names[self.file_idx])).readlines()
        smile_list = smile_list[:]
        print("Converting smiles...")
        graphs = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(self.filtered_to_nx)(s) for s in tqdm(smile_list)
        )
        graphs = [g for g in graphs if (g is not None)]
        return graphs

class PolymerDataModule(MolecularDataModule):
    def __init__(self, cfg, guidance=False):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        self.graph_type = cfg.dataset.graph_type

        self.reduction_args = get_red_arg_dict(cfg)

        target = getattr(cfg.guidance, 'guidance_target', None)
        transform = RemoveYTransform()
        if guidance:
            if target == 'logp':
                transform = SelectLogPTransform()
            else:
                transform = None
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            'train': PolymerDataset(stage='train', root=root_path,
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args),
            'val': PolymerDataset(stage='val', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args),
            'test': PolymerDataset(stage='test', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform,
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args)}
        super().__init__(cfg, datasets)


class CoarsePolymerInfos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.name = 'coarsed_Polymer'
        super().__init__(cfg)


class ExpandPolymerInfos(AbstractExpandDatasetInfos):
    def __init__(self, cfg):
        self.name = 'expanded_Polymer'

        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.max_weight = 1000
        self.valencies = [4, 3, 2, 1, 3, 2, 4]
        self.max_valencies = [4, 3, 2, 1, 5, 6, 4] # need to fill this if you want to contrain scaffold
        self.bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        {'C': 0, 'N': 1, 'O':2,  'F': 3, 'P': 4, 'S':5, 'Si': 6}
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 30.07, 6: 32.065, 7: 78.97}
        self.n_nodes = torch.tensor([0, 0, 0, 0, 0, 1.30970623e-05, 1.30970623e-05, 2.61941247e-05, 2.61941247e-05,
        7.85823740e-05, 1.17873561e-04, 1.04776499e-04, 1.96455935e-04,
        2.75038309e-04, 4.06008932e-04, 5.10785431e-04, 9.03697301e-04,
        9.16794363e-04, 1.17873561e-03, 1.13944442e-03, 1.32280330e-03,
        1.67642398e-03, 2.89445077e-03, 3.45762445e-03, 4.78042775e-03,
        5.77580449e-03, 7.46532553e-03, 7.72726677e-03, 1.02811939e-02,
        1.04252616e-02, 1.41972156e-02, 1.32935183e-02, 1.82180137e-02,
        1.65546868e-02, 2.20685500e-02, 1.96193994e-02, 2.56571451e-02,
        2.17542205e-02, 2.79491310e-02, 2.20816471e-02, 3.00839522e-02,
        2.31687033e-02, 3.15639202e-02, 2.28150826e-02, 3.05816405e-02,
        2.10731733e-02, 2.97958168e-02, 1.89514492e-02, 2.84861106e-02,
        1.83096931e-02, 2.75562191e-02, 1.61617749e-02, 2.54999804e-02,
        1.50092334e-02, 2.35747122e-02, 1.39090802e-02, 2.34175474e-02,
        1.26779563e-02, 2.22781030e-02, 1.23374327e-02, 2.12303380e-02,
        1.12110854e-02, 2.00777966e-02, 1.12765707e-02, 1.88597698e-02,
        1.05562322e-02, 1.62010661e-02, 9.92757325e-03, 1.62796485e-02,
        8.80122589e-03, 1.44853509e-02, 8.19876102e-03, 1.34899742e-02,
        7.05931660e-03, 1.15254148e-02, 6.03774573e-03, 1.03859704e-02,
        5.23882493e-03, 9.03697301e-03, 4.13867170e-03, 8.02849921e-03,
        3.44452739e-03, 6.39136642e-03, 2.54083009e-03, 5.17333962e-03,
        2.48844184e-03, 4.36132176e-03, 1.72881223e-03, 3.53620683e-03,
        1.45377392e-03, 3.19568321e-03, 1.11325030e-03, 2.18720941e-03,
        6.67950179e-04, 1.82049166e-03, 8.64406114e-04, 1.46687098e-03,
        3.40523621e-04, 1.02157086e-03, 2.88135371e-04, 8.90600238e-04,
        1.96455935e-04, 4.97688368e-04, 5.23882493e-05, 4.71494244e-04,
        1.57164748e-04, 2.09552997e-04, 6.54853116e-05, 2.35747122e-04,
        1.83358873e-04, 1.30970623e-04, 3.92911870e-05, 9.16794363e-05,
        1.30970623e-05, 1.30970623e-04, 1.30970623e-05, 2.61941247e-05,
        5.23882493e-05, 2.61941247e-05, 1.30970623e-05, 1.30970623e-05])

        super().__init__(cfg)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0: 7] = torch.tensor([0.0000, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053])

if __name__ == '__main__':
    import yaml
    from omegaconf import DictConfig
    with open('/data2/chensm22/HRS/MHdiff/configs/guacamol_exp/e_custom.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    PolymerDataModule(cfg=DictConfig(cfg))
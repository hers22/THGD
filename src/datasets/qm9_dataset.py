import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch_geometric
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

from src.analysis.rdkit_functions import calculate_base_properties, penalized_property

import src.utils as utils
from src.datasets.dataset_utils import (to_list,
                                        to_nx,
                                        files_exist,
                                        RemoveYTransform,
                                        SelectHOMOTransform,
                                        SelectMuTransform)
# from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
# from src.analysis.rdkit_functions import compute_molecular_metrics

from src.datasets.abstract_dataset import get_red_arg_dict, AbstractMolecularDataset, MolecularDataModule, AbstractExpandDatasetInfos, AbstractDatasetInfos


atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
atom_decoder = ['H', 'C', 'N', 'O', 'F']
bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor([1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
                           1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.])




class QM9Dataset(AbstractMolecularDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    atom_decoder = ['H', 'C', 'N', 'O', 'F']
    bonds = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

    def __init__(self, stage, root, remove_h: bool, graph_type='expand',
                 transform=None, pre_transform=None, reduction_args={}):
        self.remove_h = remove_h
        super().__init__(stage=stage, root=root, graph_type=graph_type, 
                         transform=transform, pre_transform=pre_transform, 
                         reduction_args=reduction_args)

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def get_nx_graphs(self):
        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)

        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        no_H_suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)
        # suppl = list(iter(suppl))[:1000]

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)
        target = torch.tensor(target_df.values, dtype=torch.float)
        target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
        target = target * conversion.view(1, -1)

        print("Converting smiles...")
        graphs = []
        for i, s in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            G = to_nx(s,self.atom_encoder, self.bonds, cal_prop=False)
            # logP, qed, sa_score = calculate_base_properties(no_H_suppl[i])
            # P_logP = penalized_property(no_H_suppl[i])
            # custom_y = torch.tensor([logP, qed, sa_score, P_logP])
            if G.number_of_nodes() > 6:
                # y = target[target_df.index.get_loc(i)]
                # y = torch.hstack((custom_y, y[:1], y[2:3]))         # mu, homo
                # G.graph['y']=y
                graphs.append(G)
        return graphs

class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg, guidance=False):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        self.graph_type = cfg.dataset.graph_type
        # self.reduce_frac = cfg.dataset.reduce_frac

        self.reduction_args = get_red_arg_dict(cfg)

        target = getattr(cfg.general, 'guidance_target', None)
        if guidance and target == 'mu':
            transform = SelectMuTransform()
        elif guidance and target == 'homo':
            transform = SelectHOMOTransform()
        elif guidance and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()
        self.guidance = guidance 

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            'train': QM9Dataset(stage='train', root=root_path,
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform if self.guidance else RemoveYTransform(),
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args),
            'val': QM9Dataset(stage='val', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform if self.guidance else RemoveYTransform(),
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args),
            'test': QM9Dataset(stage='test', root=root_path, 
                                    graph_type=cfg.dataset.graph_type,
                                    transform=transform if self.guidance else RemoveYTransform(),
                                    remove_h=cfg.dataset.remove_h,
                                    reduction_args=self.reduction_args)}
        super().__init__(cfg, datasets)


class CoarseQM9Infos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.name = 'coarsed_qm9'
        super().__init__(cfg)


class ExpandQM9Infos(AbstractExpandDatasetInfos):
    def __init__(self, cfg):
        self.remove_h = cfg.dataset.remove_h

        self.name = 'expanded_qm9'
        self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
        self.valencies = [1, 4, 3, 2, 1]
        self.num_atom_types = 5
        self.max_n_nodes = 29
        self.max_weight = 390
        self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}

        self.n_nodes = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                        9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                        1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                        1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                        5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

        super().__init__(cfg)
        # super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0:6] = torch.tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

if __name__ == '__main__':
    import yaml
    from omegaconf import DictConfig
    with open('/data2/chensm22/HRS/MHdiff/configs/dataset/masked_qm9.yaml', 'r') as file:
        cfg = {'dataset':yaml.safe_load(file),
               'train':{'batch_size':32, 'num_workers': 4},
               'general':{'name':'test'}
               }
    
    QM9DataModule(cfg=DictConfig(cfg))
from src.analysis.rdkit_functions import check_stability
from src.datasets.qm9_dataset import ExpandQM9Infos
import yaml
from omegaconf import DictConfig


with open('/data2/chensm22/HRS/MHdiff/configs/qm9_exp/e_custom.yaml') as f:
    cfg = yaml.safe_load(f)
cfg = DictConfig(cfg)
dataset_info = ExpandQM9Infos(cfg)

atom_types = 
edge_types = 
atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
atom_decoder = ['H', 'C', 'N', 'O', 'F']
edge_types = {None:0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
check_stability(atom_types, edge_types, dataset_info=None, debug=False,atom_decoder=None)
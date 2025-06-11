import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb
import torch.nn.functional as F
import torch.distributed as dist



def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)

def has_edge_masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, has_edge, batch):
    X_type_mask, _ = to_dense_batch(x=x_type_mask, batch=batch, fill_value=0)

    masked_edge_index, edge_type_mask = torch_geometric.utils.remove_self_loops(masked_edge_index, edge_type_mask)
    max_num_nodes = X_type_mask.shape[1]
    E_type_mask = to_dense_adj(edge_index=masked_edge_index, batch=batch, edge_attr=edge_type_mask, max_num_nodes=max_num_nodes)
    return X_type_mask, E_type_mask

def masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, batch, return_node_mask=False):
    X_type_mask, node_mask = to_dense_batch(x=x_type_mask, batch=batch, fill_value=0)

    # masked_edge_index, edge_type_mask = torch_geometric.utils.remove_self_loops(masked_edge_index, edge_type_mask)
    max_num_nodes = X_type_mask.shape[1]
    E_type_mask = to_dense_adj(edge_index=masked_edge_index, batch=batch, edge_attr=edge_type_mask, max_num_nodes=max_num_nodes)
    if return_node_mask:
        return X_type_mask, E_type_mask, node_mask
    else:
        return X_type_mask, E_type_mask

def coarsed_to_dense(x, x_aug, edge_index, edge_attr, batch):

    X = torch.cat([x,x_aug],-1)
    X, node_mask = to_dense_batch(x=X, batch=batch)
    max_num_nodes = X.size(1)
    
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X[..., :x.shape[-1]], X_aug=X[..., x.shape[-1]:], E=E, y=None), node_mask

def to_dense(x, edge_index, edge_attr, batch, x_classes=None, e_classes=None, remove_loop=True):
    if x_classes is not None:
        x = F.one_hot(x,num_classes=x_classes).float()
    if e_classes is not None:
        edge_attr = F.one_hot(edge_attr,num_classes=e_classes).float()
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    if remove_loop:
        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E, remove_loop=remove_loop)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E, remove_loop=True):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    if remove_loop:
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model
    saved_dataset = saved_cfg.dataset

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)
    
    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.dataset, True)
        with open_dict(cfg.dataset):
            if key not in cfg.dataset.keys():
                setattr(cfg.dataset, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y, X_aug=None):
        self.X = X
        self.E = E
        self.y = y
        self.X_aug = X_aug

    def __str__(self) -> str:
        return f"shape of X: {self.X.shape} \n" + \
                f"shape of E: {self.E.shape} \n" + \
                f"shape of y: {self.y.shape} \n" + \
               ( f"shape of X_aug: {self.X_aug.shape} \n" if self.X_aug is not None else "")

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if self.X_aug is not None: self.X_aug = self.X_aug.type_as(x)
        return self
    
    def to(self, device):
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        if self.X_aug is not None: self.X_aug = self.X_aug.to(device)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1

            if self.X_aug is not None: 
                self.X_aug = torch.argmax(self.X_aug, dim=-1)
                self.X_aug[node_mask == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if self.X_aug is not None: self.X_aug = self.X_aug * x_mask
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
    
    def type_mask(self, X_type_mask, E_type_mask, encode_no_edge=False):
        X_null_type_mask = (X_type_mask == 0).bool()
        E_null_type_mask = (E_type_mask == 0).bool()
        self.X = self.X * ~X_null_type_mask.unsqueeze(-1)
        self.E = self.E * ~E_null_type_mask.unsqueeze(-1)
        if encode_no_edge:
            self.E[:,:,:,0] = E_null_type_mask # set no edge of E where null mask is True
        return self

def sync_params_across_gpus(model):
    for param in model:
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')



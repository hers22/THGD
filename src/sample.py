import warnings
import argparse
import pickle
import yaml
import hydra
from omegaconf import OmegaConf

import time

import torch
import pytorch_lightning as pl
torch.cuda.empty_cache()

from omegaconf import DictConfig
from torch_geometric.utils import to_dense_adj
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.metrics.molecular_metrics import SamplingMolecularMetrics
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization, NonMolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

from src.coarse_diffusion_model_discrete import CoarseDiscreteDenoisingDiffusion
from src.refinement_diffusion_model_discrete import RefinementDiscreteDenoisingDiffusion
from src.datasets.dataset_utils import coarsed_sample_loader, refinement_sample_loader
import os


warnings.filterwarnings("ignore", category=PossibleUserWarning)

pwd = os.path.abspath('.')

def load_coarsed_model(sample_cfg, cfg):
    dataset_config = cfg.dataset
    if dataset_config['name'] == 'coarsed_mose':
        from src.datasets import mose_dataset
        datamodule = mose_dataset.MoseDataModule(cfg)
        dataset_infos = mose_dataset.CoarseMoseInfos(cfg)
    if dataset_config['name'] == 'coarsed_guacamol':
        from src.datasets import guacamol_dataset
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.CoarseGuacamolInfos(cfg)
    if dataset_config['name'] == 'coarsed_zinc250k':
        from src.datasets import zinc250k_dataset
        datamodule = zinc250k_dataset.ZINC250kDataModule(cfg)
        dataset_infos = zinc250k_dataset.CoarseZINC250kInfos(cfg)


    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = None
    visualization_tools = NonMolecularVisualization()
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 
                        }
    model = CoarseDiscreteDenoisingDiffusion.load_from_checkpoint(
        os.path.join(pwd,sample_cfg.sample.coarse_ckpt), 
        **model_kwargs, 
        strict=False)

    OmegaConf.set_struct(model.cfg.general, False)
    model.cfg.train.batch_size = sample_cfg.sample.batch_size
    OmegaConf.set_struct(model.cfg.general, True)
    return model, dataset_infos

def load_refinement_model(sample_cfg, cfg):
    dataset_config = cfg["dataset"]
    if dataset_config['name'] == 'expanded_mose':
        from src.datasets import mose_dataset
        datamodule = mose_dataset.MoseDataModule(cfg)
        dataset_infos = mose_dataset.ExpandMoseInfos(cfg)

    if dataset_config['name'] == 'expanded_guacamol':
        from src.datasets import guacamol_dataset
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.ExpandGuacamolInfos(cfg)

    if dataset_config['name'] == 'expanded_zinc250k':
        from src.datasets import zinc250k_dataset
        datamodule = zinc250k_dataset.ZINC250kDataModule(cfg)
        dataset_infos = zinc250k_dataset.ExpandZINC250kInfos(cfg)

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, None)
    visualization_tools = MolecularVisualization(None, dataset_infos=dataset_infos)
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 
                        }
    model = RefinementDiscreteDenoisingDiffusion.load_from_checkpoint(
        os.path.join(pwd,sample_cfg.sample.refine_ckpt), 
        **model_kwargs, 
        strict=False)

    OmegaConf.set_struct(model.cfg.general, False)
    model.cfg.general.disable_tqdm = not sample_cfg.sample.enable_progress_bar
    model.cfg.train.batch_size = sample_cfg.sample.batch_size
    OmegaConf.set_struct(model.cfg.general, True)
    return model, dataset_infos

def sample_coarsed_graph(coarsed_model, device, coarsed_loader,
                        enable_progress_bar=True, 
                        save_graphs=True):
    
    coarsed_trainer = pl.Trainer(
                    strategy="ddp",  # Needed to load old checkpoints
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=[device],
                    enable_progress_bar=enable_progress_bar)
    

    start = time.time()
    coarsed_graphs = []
    _coarsed_graphs = coarsed_trainer.predict(coarsed_model, dataloaders=coarsed_loader)
    for graphs in _coarsed_graphs:
        coarsed_graphs.extend(graphs)

    print(f'Sampling Coarsed Graphs took {time.time() - start:.2f} seconds\n')

    if save_graphs:
        torch.save(coarsed_graphs, './coarsed_graphs.th')
    return coarsed_graphs

def sample_refined_graph(refinement_model, device, refine_loader,
                        enable_progress_bar=True, 
                        save_graphs=True):
    refinement_trainer = pl.Trainer(
                      strategy="ddp", 
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=[device],
                      enable_progress_bar=enable_progress_bar)
    

    start = time.time()
    refinement_graphs = []
    _refinement_graphs = refinement_trainer.predict(refinement_model, dataloaders=refine_loader)
    for graphs in _refinement_graphs:
        refinement_graphs.extend(graphs)
    

    print(f'Sampling Refine Graphs took {time.time() - start:.2f} seconds\n')

    if save_graphs:
        torch.save(refinement_graphs, './refinement_graphs.th')
    return refinement_graphs

@hydra.main(version_base='1.3', config_path='../configs', config_name='sample.yaml')
def main(sample_cfg: DictConfig):
    torch.set_float32_matmul_precision(sample_cfg.sample.precision)
    
    assert sample_cfg.sample.coarse_cfg is not None
    assert sample_cfg.sample.refine_cfg is not None

    assert sample_cfg.sample.coarse_ckpt is not None
    assert sample_cfg.sample.refine_ckpt is not None
    
    import os
    
    with open(os.path.join(pwd,sample_cfg.sample.coarse_cfg), 'r') as file:
        coarse_cfg = yaml.safe_load(file)
    coarse_cfg = DictConfig(coarse_cfg)
    coarse_cfg.train.batch_size = sample_cfg.sample.batch_size

    with open(os.path.join(pwd,sample_cfg.sample.refine_cfg), 'r') as file:
        refine_cfg = yaml.safe_load(file)
    refine_cfg = DictConfig(refine_cfg)
    refine_cfg.train.batch_size = sample_cfg.sample.batch_size
    print("OK there")
    sample(sample_cfg, coarse_cfg, refine_cfg)

def scaffold_coarsening(scaffold:str, 
                        coarsed_dataset_infos:DictConfig, 
                        refinement_dataset_infos:DictConfig):
    from src.datasets.dataset_utils import graph_from_scaffold

    coarsed_scaffold_data, expanded_scaffold_data = graph_from_scaffold(scaffold, 
                                                    coarsed_dataset_infos,
                                                    refinement_dataset_infos,
                                                    )
    coarsed_graphs = [(coarsed_scaffold_data.x, 
                        coarsed_scaffold_data.x_aug, 
                        torch.zeros([1,1], dtype=torch.long, device=coarsed_scaffold_data.x.device) if \
                        coarsed_scaffold_data.edge_index.numel() < 2 else \
                        to_dense_adj(
                            edge_index=coarsed_scaffold_data.edge_index.long(),
                            edge_attr=coarsed_scaffold_data.edge_attr.long())[0]
                        )
                    ]
    return coarsed_graphs, coarsed_scaffold_data, expanded_scaffold_data

def sample(sample_cfg: DictConfig, coarse_cfg: DictConfig, refine_cfg: DictConfig):
    # load coarsed model
    coarsed_model, coarsed_dataset_infos = load_coarsed_model(sample_cfg, coarse_cfg)

    # load refinement model
    refinement_model, refinement_dataset_infos = load_refinement_model(sample_cfg, refine_cfg)


    # check scaffolds
    coarsed_scaffold_data = None
    expanded_scaffold_data = None
    scaffold = sample_cfg.sample.scaffold
    if scaffold is not None:
        coarsed_graphs, coarsed_scaffold_data, expanded_scaffold_data = scaffold_coarsening(scaffold, 
                                                                                coarsed_dataset_infos,
                                                                                refinement_dataset_infos)

    # coarsed graph sample
    coarsed_loader = coarsed_sample_loader(
                                sample_cfg.sample.num_batch, 
                                sample_cfg.sample.num_nodes, 
                                coarsed_scaffold_data)

    coarsed_graphs = sample_coarsed_graph(coarsed_model, 
                                          sample_cfg.sample.devices, 
                                          coarsed_loader,
                                        enable_progress_bar=sample_cfg.sample.enable_progress_bar, 
                                        save_graphs=sample_cfg.sample.save_graphs)

    # refinement graph sample
    refine_loader = refinement_sample_loader(
                                    sample_cfg.sample.num_batch, 
                                    sample_cfg.sample.batch_size, 
                                    coarsed_graphs,
                                    expanded_scaffold=expanded_scaffold_data,
                                    )
    refinement_graphs = sample_refined_graph(refinement_model, 
                                             sample_cfg.sample.devices, 
                                             refine_loader,
                                            enable_progress_bar=sample_cfg.sample.enable_progress_bar, 
                                            save_graphs=sample_cfg.sample.save_graphs)
        

if __name__ == '__main__':
    main()


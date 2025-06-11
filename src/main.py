# import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

# from coarsed_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.coarse_diffusion_model_discrete import CoarseDiscreteDenoisingDiffusion
from THGD.src.refinement_diffusion_model_discrete import ExpandDiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
warnings.filterwarnings("ignore", category=UserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.dateset.graph_type == 'coarse':
        model = CoarseDiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = ExpandDiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg["dataset"]['graph_type'] == 'coarse':
        model = CoarseDiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = ExpandDiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        OmegaConf.set_struct(new_cfg[category], False)
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]
        OmegaConf.set_struct(new_cfg[category], True)

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['coarsed_guacamol', 'expanded_guacamol', 
                                  'coarsed_mose', 'expanded_mose', 
                                  'coarsed_qm9', 'expanded_qm9',
                                  'coarsed_zinc250k', 'expanded_zinc250k',
                                  'coarsed_polymers', 'expanded_polymers']:
        print(dataset_config.graph_type)
        if dataset_config.graph_type == 'coarse':
            from src.analysis.visualization import NonMolecularVisualization

            if dataset_config['name'] == 'coarsed_qm9':
                from src.datasets import qm9_dataset
                datamodule = qm9_dataset.QM9DataModule(cfg)
                dataset_infos = qm9_dataset.CoarseQM9Infos(cfg)

            if dataset_config['name'] == 'coarsed_mose':
                from src.datasets import mose_dataset
                datamodule = mose_dataset.MoseDataModule(cfg)
                dataset_infos = mose_dataset.CoarseMoseInfos(cfg)
            
            if dataset_config['name'] == 'coarsed_guacamol':
                from src.datasets import guacamol_dataset
                datamodule = guacamol_dataset.GuacamolDataModule(cfg)
                dataset_infos = guacamol_dataset.CoarseGuacamolInfos(cfg)

            if dataset_config['name'] == 'coarsed_zinc250k':
                from src.datasets import zinc_dataset
                datamodule = zinc_dataset.ZINC250kDataModule(cfg)
                dataset_infos = zinc_dataset.CoarseZINC250kInfos(cfg)

            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = DummyExtraFeatures()
                
            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                            domain_features=domain_features)
            train_metrics = TrainAbstractMetricsDiscrete()
            sampling_metrics = None
            visualization_tools = NonMolecularVisualization()

        elif dataset_config.graph_type == 'expand':
            from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
            from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
            from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
            from src.analysis.visualization import MolecularVisualization

            if dataset_config['name'] == 'expanded_mose':
                from src.datasets import mose_dataset
                datamodule = mose_dataset.MoseDataModule(cfg)
                dataset_infos = mose_dataset.ExpandMoseInfos(cfg)

            if dataset_config['name'] == 'expanded_qm9':
                from src.datasets import qm9_dataset
                datamodule = qm9_dataset.QM9DataModule(cfg)
                dataset_infos = qm9_dataset.ExpandQM9Infos(cfg)

            if dataset_config['name'] == 'expanded_guacamol':
                from src.datasets import guacamol_dataset
                datamodule = guacamol_dataset.GuacamolDataModule(cfg)
                dataset_infos = guacamol_dataset.ExpandGuacamolInfos(cfg)

            if dataset_config['name'] == 'expanded_zinc250k':
                from src.datasets import zinc_dataset
                datamodule = zinc_dataset.ZINC250kDataModule(cfg)
                dataset_infos = zinc_dataset.ExpandZINC250kInfos(cfg)


            if cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                            domain_features=domain_features)
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
            train_smiles = None
            # We do not evaluate novelty during training
            sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
            # sampling_metrics = None
            visualization_tools = MolecularVisualization(None, dataset_infos=dataset_infos)
        else: raise NotImplementedError

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 
                        }
    else: raise NotImplementedError

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.dataset.graph_type == 'coarse':
        model = CoarseDiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    elif cfg.dataset.graph_type == 'expand':
        model = ExpandDiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                    #   accelerator='cpu',
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.train.devices,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.general.wandb == 'diasbled',
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

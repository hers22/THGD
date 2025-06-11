import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from tqdm.auto import tqdm
import torch.distributed as dist

from src.models.transformer_model import GraphTransformer
# from models.transformer_model_type_mask import GraphTransformer

from src.diffusion.noise_schedule import DDIMNoiseScheduleDiscrete, DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition, MaskedMarginalUniformTransition
from src.diffusion import expand_diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils


class RefinementDiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = DDIMNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps,
            skip=self.cfg.general.skip if hasattr(self.cfg.general,'skip') else 2,
        )

        # mask transition
        masked_node_distribution = dataset_infos.node_types
        masked_edge_distribution = dataset_infos.edge_types
        num_masks, _ = masked_node_distribution.shape
        self.num_masks = num_masks + 1
        print("Masked node distribution:")
        print(masked_node_distribution) # num_masks x num_categories
        print("Masked edge distribution:")
        print(masked_edge_distribution)

        eps = 1e-12
        null_mask_marginals = torch.ones(1, self.Xdim_output)
        masked_node_distribution = torch.cat([null_mask_marginals, masked_node_distribution],0) + eps
        null_mask_marginals = torch.ones(1, self.Edim_output)
        masked_edge_distribution = torch.cat([null_mask_marginals, masked_edge_distribution],0) + eps  # n_mask + 1, n_class
        masked_node_distribution = masked_node_distribution / masked_node_distribution.sum(-1,keepdim=True)
        masked_edge_distribution = masked_edge_distribution / masked_edge_distribution.sum(-1,keepdim=True)


        self.transition_model = MaskedMarginalUniformTransition(masked_node_distribution=masked_node_distribution, masked_edge_distribution=masked_edge_distribution,
                                                            y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=masked_node_distribution, E=masked_edge_distribution,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)
        self.limit_dist = self.limit_dist.to(self.device)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        x_type_mask=data.x_type_mask + 1 # encode null mask
        masked_edge_index=data.masked_edge_index
        edge_type_mask=data.edge_type_mask + 1 # encode null mask
        batch = data.batch
        
        x_target=data.x_target
        edge_target=data.edge_target
        edge_exist_mask = edge_target!=0
        edge_target = edge_target[edge_exist_mask]
        edge_index = masked_edge_index[:,edge_exist_mask]

        X_type_mask, E_type_mask = utils.masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, batch)
        dense_data, node_mask = utils.to_dense(x_target, edge_index, edge_target, batch, x_classes=self.Xdim_output, e_classes=self.Edim_output)

        dense_data = dense_data.type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, X_type_mask, E_type_mask, node_mask)
    
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        pred.mask(node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        # 定义优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay
        )
        
        return {
            "optimizer": optimizer,
        }

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        # print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        x_type_mask=data.x_type_mask + 1 # encode null mask
        masked_edge_index=data.masked_edge_index 
        edge_type_mask=data.edge_type_mask + 1 # encode null mask
        batch = data.batch
        
        x_target=data.x_target
        edge_target=data.edge_target
        edge_exist_mask = edge_target!=0
        edge_target = edge_target[edge_exist_mask]
        edge_index = masked_edge_index[:,edge_exist_mask]

        X_type_mask, E_type_mask = utils.masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, batch)
        dense_data, node_mask = utils.to_dense(x_target, edge_index, edge_target, batch, x_classes=self.Xdim_output, e_classes=self.Edim_output)

        dense_data = dense_data.type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, X_type_mask, E_type_mask, node_mask)
    
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        pred.mask(node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, X_type_mask, E_type_mask, data.y,  node_mask)
        
        if i == 0:
            self.val_X_type_mask = X_type_mask
            self.val_E_type_mask = E_type_mask
            self.val_node_mask = node_mask
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4],
                    #    "optim/lr":current_lr
                       }, commit=False)

        # self.print(f"Current Learning Rate: {current_lr:.6f}")
        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))


        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.masked_sample_batch(
                                                batch_id=ident,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 resample_step=self.cfg.general.resample_step,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")
        del self.val_X_type_mask
        del self.val_E_type_mask

    def on_predict_epoch_start(self):
        self.print("Starting Sampling...")
        self.start_sample_time = time.time()

    def predict_step(self, batch, batch_idx):
        samples_left_to_save = self.cfg.general.samples_to_save
        chains_left_to_save = self.cfg.general.chains_to_save
        number_chain_steps = self.cfg.general.number_chain_steps
        bs = self.cfg.train.batch_size
        resample_step = self.cfg.general.resample_step
        to_save = min(samples_left_to_save, bs)
        chains_save = min(chains_left_to_save, bs)

        if isinstance(batch, dict):
            expanded_batch = batch['expanded_batch']
            scaffold_X = batch['scaffold_X']
            scaffold_E = batch['scaffold_E']
            scaffold_X_type_mask = batch['scaffold_X_type_mask']
            scaffold_E_type_mask = batch['scaffold_E_type_mask']
            scaffold_attrs = [scaffold_X, scaffold_E]
        else:
            expanded_batch = batch
            scaffold_attrs = []

        x_type_mask=expanded_batch.x_type_mask + 1 # encode null mask
        masked_edge_index=expanded_batch.masked_edge_index
        edge_type_mask=expanded_batch.edge_type_mask + 1 # encode null mask
        batch = expanded_batch.batch
        X_type_mask, E_type_mask, node_mask = \
            utils.masks_to_dense(x_type_mask, masked_edge_index, edge_type_mask, batch, True)
        if isinstance(batch, dict):
            n_constrain = scaffold_X_type_mask.shape[1]
            X_type_mask[:, :n_constrain] = scaffold_X_type_mask
            E_type_mask[:, :n_constrain, :n_constrain] = scaffold_E_type_mask

        samples = self.masked_sample_batch(
                                        batch_id=batch_idx,
                                        save_final=to_save, 
                                        keep_chain=chains_save, 
                                        number_chain_steps=number_chain_steps,
                                        X_type_mask=X_type_mask, 
                                        E_type_mask=E_type_mask, 
                                        node_mask=node_mask,
                                        resample_step=resample_step,
                                        scaffold_attrs=scaffold_attrs
                                        )
        if hasattr(self, 'sample_list'):
            self.sample_list.append(samples)
        else:
            self.sample_list = [samples]
        return samples
    
    def on_predict_epoch_end(self):
        samples = []
        for sample in self.sample_list:
            samples.extend(sample)
        self.sampling_metrics.forward(samples, 'sample_expand', 0, 0, test=True, local_rank=self.local_rank)
        print(f'Sampling Expanded Graphs took {time.time() - self.start_sample_time:.2f} seconds\n')
        return samples

    def kl_prior(self, X, E, X_type_mask, E_type_mask, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        X_type_mask_expanded = X_type_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Xdim_output, self.Xdim_output)
        Qtb_X = torch.gather(Qtb.X, 1, X_type_mask_expanded) # (bs, n, dx_out, dx_out)

        bs, n, _ = E_type_mask.shape
        E_type_mask_expanded = E_type_mask.reshape(bs, n*n)
        E_type_mask_expanded = E_type_mask_expanded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Edim_output, self.Edim_output)
        Qtb_E = torch.gather(Qtb.E, 1, E_type_mask_expanded) # (bs, n, n, de_out, de_out)
        Qtb_E = Qtb_E.reshape(bs, n, n, self.Edim_output, self.Edim_output)
        # Compute transition probabilities
        # probX = X @ Qtb_X  # (bs, n, dx_out)
        # probE = E @ Qtb_E  # (bs, n, n, de_out)
        probX = (X.unsqueeze(-2) @ Qtb_X).squeeze()  # (bs, n, dx_out)
        probE = (E.unsqueeze(-2) @ Qtb_E).squeeze()  # (bs, n * n, de_out)
        assert probX.shape == X.shape

        bs, n_max = node_mask.shape
        x_limit = self.limit_dist.X.to(X_type_mask.device)
        x_limit = x_limit[X_type_mask.flatten()]
        x_limit = x_limit.reshape(bs, n_max, -1)
        e_limit = self.limit_dist.E.to(X_type_mask.device)
        e_limit = e_limit[E_type_mask.flatten()]
        e_limit = e_limit.reshape(bs, n_max, n_max, -1)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = expand_diffusion_utils.mask_distributions(true_X=x_limit.clone(),
                                                                                      true_E=e_limit.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      X_type_mask=X_type_mask, 
                                                                                      E_type_mask=E_type_mask,
                                                                                      node_mask=node_mask)
        
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return expand_diffusion_utils.sum_except_batch(kl_distance_X) + \
               expand_diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, X_type_mask, E_type_mask, y, pred, noisy_data, node_mask):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = expand_diffusion_utils.masked_posterior_distributions(X=X, E=E, X_type_mask=X_type_mask, E_type_mask=E_type_mask, 
                                                            y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = expand_diffusion_utils.masked_posterior_distributions(X=pred_probs_X, E=pred_probs_E, X_type_mask=X_type_mask, E_type_mask=E_type_mask,
                                                            y=pred_probs_y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))
        assert ~(prob_pred.E.isnan().any())
        assert ~(prob_true.E.isnan().any())

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = expand_diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                X_type_mask=X_type_mask, 
                                                                                                E_type_mask=E_type_mask,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = self.val_X_kl(prob_true.X, torch.log(prob_pred.X))
        kl_e = self.val_E_kl(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, X_type_mask, E_type_mask, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)


        X_type_mask_expanded = X_type_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Xdim_output, self.Xdim_output)
        Q0_X = torch.gather(Q0.X, 1, X_type_mask_expanded) # (bs, n, dx_out, dx_out)
        probX0 = (X.unsqueeze(-2) @ Q0_X).squeeze()  # (bs, n, dx_out)

        bs, n, _ = E_type_mask.shape
        E_type_mask_expanded = E_type_mask.reshape(-1, n*n)
        E_type_mask_expanded = E_type_mask_expanded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Edim_output, self.Edim_output)
        Q0_E = torch.gather(Q0.E, 1, E_type_mask_expanded) # (bs, n, n, de_out, de_out)
        Q0_E = Q0_E.reshape(-1, n, n, self.Edim_output, self.Edim_output)
        probE0 = (E.unsqueeze(-2) @ Q0_E).squeeze()  # (bs, n, n, de_out)

        # probX0 = X @ Q0.X  # (bs, n, dx_out)
        # probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = expand_diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, X_type_mask=X_type_mask, E_type_mask=E_type_mask, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0)
        sampled_0.type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        sampled_0.mask(node_mask)
        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      'X_type_mask':X_type_mask,'E_type_mask':E_type_mask,'t': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)
        pred0.type_mask(noisy_data['X_type_mask'],noisy_data['E_type_mask'],encode_no_edge=False)
        pred0.mask(node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, X_type_mask, E_type_mask, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, n_mask, dx_in, dx_out), (bs, n_mask, de_in, de_out)
        # assert (abs(Qtb.X.sum(dim=-1) - 1.) < 1e-4).all(), Qtb.X.sum(dim=-1) - 1
        # assert (abs(Qtb.E.sum(dim=-1) - 1.) < 1e-4).all()

        bs, n = X_type_mask.shape
        # Compute transition probabilities
        X_type_mask_expanded = X_type_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Xdim_output, self.Xdim_output)
        Qtb_X = torch.gather(Qtb.X, 1, X_type_mask_expanded) # (bs, n, dx_out, dx_out)
        probX = (X.unsqueeze(-2) @ Qtb_X).squeeze(-2)  # (bs, n, dx_out)

        E_type_mask_expanded = E_type_mask.reshape(bs, n * n)
        E_type_mask_expanded = E_type_mask_expanded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Edim_output, self.Edim_output)
        Qtb_E = torch.gather(Qtb.E, 1, E_type_mask_expanded) # (bs, n, n, de_out, de_out)
        Qtb_E = Qtb_E.reshape(bs, n, n, self.Edim_output, self.Edim_output)
        probE = (E.unsqueeze(-2) @ Qtb_E).squeeze(-2)  # (bs, n * n, de_out)

        sampled_t = expand_diffusion_utils.sample_discrete_features(probX=probX, probE=probE, X_type_mask=X_type_mask, E_type_mask=E_type_mask, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t)
        z_t = z_t.type_mask(X_type_mask=X_type_mask, E_type_mask=E_type_mask, encode_no_edge=False)
        z_t = z_t.mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask,
                      'X_type_mask': X_type_mask, 'E_type_mask': E_type_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, X_type_mask, E_type_mask, y, node_mask):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        # log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, X_type_mask, E_type_mask, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, X_type_mask, E_type_mask, y, pred, noisy_data, node_mask)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, X_type_mask, E_type_mask, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        # nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        nlls = kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = self.val_nll(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                    #    "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def masked_sample_batch(self, batch_id:int, keep_chain: int, number_chain_steps: int, save_final: int, 
                     resample_step=1, X_type_mask=None, E_type_mask=None, node_mask=None, scaffold_attrs=[]):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if node_mask is None:
            node_mask = self.val_node_mask
            X_type_mask = self.val_X_type_mask
            E_type_mask = self.val_E_type_mask
        else:
            node_mask = node_mask
            X_type_mask = X_type_mask
            E_type_mask = E_type_mask

        batch_size = node_mask.shape[0]
        n_nodes = node_mask.long().sum(-1)
        n_max = torch.max(n_nodes).item()
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = expand_diffusion_utils.sample_masked_discrete_feature_noise(self.limit_dist, X_type_mask, E_type_mask, self.num_masks, node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        time_range = torch.arange(0, self.T, self.cfg.general.skip)
        for s_int in tqdm(reversed(time_range), 
                            total=self.T // self.cfg.general.skip, 
                            disable=True if not hasattr(self.cfg.general, 'disable_tqdm') else self.cfg.general.disable_tqdm):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            for _ in range(resample_step):
                sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, X_type_mask, E_type_mask, node_mask, scaffold_attrs)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.type_mask(X_type_mask,E_type_mask,encode_no_edge=True)
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        sampled_s = self.mask_scaffold(scaffold_attrs, sampled_s)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = expand_diffusion_utils.reverse_tensor(chain_X)
            chain_E = expand_diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                print('saving chain path:', result_path)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    try:
                        _ = self.visualization_tools.visualize_chain(result_path,
                                                                    chain_X[:, i, :].numpy(),
                                                                    chain_E[:, i, :].numpy())
                        print('\r{}/{} complete'.format(i+1, num_molecules), end='')
                    except:
                        print('\r{}/{} failed'.format(i+1, num_molecules), end='')
            print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            print("Done.")

        return molecule_list

    def mask_scaffold(self, scaffold_attrs, sampled_s):
        try:
            if not len(scaffold_attrs)==0 and scaffold_attrs[0]:
                # scaffold extension mask operation
                scaffold_X, scaffold_E = scaffold_attrs
                n_nodes_scaffold = scaffold_X.shape[0]

                sampled_s.X[:, :n_nodes_scaffold] = scaffold_X
                sampled_s.E[:, :n_nodes_scaffold, :n_nodes_scaffold] = scaffold_E
        finally:
            return sampled_s

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, X_type_mask, E_type_mask, node_mask, scaffold_attrs, check_validity=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_normalized=t, skip=True)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s, skip=True)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t, skip=True)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask, 
                      'X_type_mask':X_type_mask, 
                      'E_type_mask':E_type_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.type_mask(noisy_data['X_type_mask'],noisy_data['E_type_mask'],encode_no_edge=False)
        pred.mask(node_mask)
        
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        p_s_and_t_given_0_X = expand_diffusion_utils.compute_masked_batched_over0_posterior_distribution(X_t=X_t, 
                                                                                                Qt=Qt.X, 
                                                                                                Qsb=Qsb.X, 
                                                                                                Qtb=Qtb.X, 
                                                                                                type_mask=X_type_mask)
        p_s_and_t_given_0_E = expand_diffusion_utils.compute_masked_batched_over0_posterior_distribution(X_t=E_t, 
                                                                                                Qt=Qt.E, 
                                                                                                Qsb=Qsb.E, 
                                                                                                Qtb=Qtb.E, 
                                                                                                type_mask=E_type_mask)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = expand_diffusion_utils.sample_discrete_features(probX=prob_X, probE=prob_E, X_type_mask=X_type_mask, E_type_mask=E_type_mask, node_mask=node_mask)

        sampled_s = self.mask_scaffold(scaffold_attrs, sampled_s)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))\
            .type_mask(X_type_mask,E_type_mask,encode_no_edge=False)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))\
            .type_mask(X_type_mask,E_type_mask,encode_no_edge=True)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        X_type_mask = F.one_hot(noisy_data['X_type_mask'], num_classes=self.num_masks)
        E_type_mask = F.one_hot(noisy_data['E_type_mask'], num_classes=self.num_masks)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X, X_type_mask), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E, E_type_mask), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

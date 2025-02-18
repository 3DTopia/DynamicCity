import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

import dynamic_city.utils.constants as C
from dynamic_city.dataset.builder import get_occ_sequence_dataloaders
from dynamic_city.utils.ckpt_utils import get_latest_ckpt
from dynamic_city.utils.data_utils import convert_voxels_to_layouts, parse_rollout_path
from dynamic_city.utils.dist_utils import distributed, func_rank_0, print_text, rank_0, wlog
from dynamic_city.utils.hexplane_utils import hexplane_to_rollout
from dynamic_city.utils.loss_utils import build_losses, calculate_ce_weight
from dynamic_city.utils.metrics import Metrics
from dynamic_city.utils.vae_train_utils import get_pred_label
from dynamic_city.vae.vae import DynamicCityAE


class VAETrainer:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device

        # dataset conf
        self.num_classes = conf.dataset.num_classes
        self.sequence_length = conf.dataset.sequence_length
        self.class_names = [value for key, value in sorted(conf.dataset.semantic_names.items())]

        # trainer conf
        self.amp = conf.trainer.amp
        self.num_epochs = conf.trainer.num_epochs
        self.log_frequency = conf.trainer.log_frequency
        self.log_train = conf.trainer.log_train
        self.log_class = conf.trainer.log_class
        self.log_frame = conf.trainer.log_frame

        # setup dataloaders
        self.train_loader, self.valid_loader = get_occ_sequence_dataloaders(conf.dataset, conf)

        # load model
        self.model = DynamicCityAE(conf).to(self.device)
        if distributed():
            self.model = DDP(self.model, device_ids=[self.device])
        print_text(f'VAE Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        # setup loss
        self.loss_history = dict()
        self.loss_names = list(conf.model.loss_weights.keys())
        self.loss_weights = dict(**conf.model.loss_weights)
        ce_weight = calculate_ce_weight(conf.dataset).to(device).float() \
            if 'ce' in self.loss_names and conf.model.ce_class_weight else None
        self.loss_functions = build_losses(conf.model, ce_weight=ce_weight)

        # setup optimizer
        self.optimizer, self.scheduler = self.setup_optimizer()
        if self.amp:
            self.scaler = GradScaler('cuda')

        # setup metrics
        self.use_valid_mask = conf.model.use_valid_mask
        self.metrics = self.setup_metrics()

        # counters
        self.current_step = 0
        self.current_epoch = 0
        self.best_valid_miou = 0.0

        # resume
        self.load_checkpoint(self.get_resume_ckpt_path())

    def setup_optimizer(self):
        # setup optimizer
        optimizer_class = dict(
            adam=Adam,
            adamw=AdamW,
        )[self.conf.model.optimizer_type]
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.conf.model.learning_rate,
            weight_decay=self.conf.model.weight_decay
        )

        # setup scheduler
        if self.conf.model.scheduler_type == 'multisteplr':
            scheduler = MultiStepLR(optimizer, self.conf.model.lr_scheduler_steps, self.conf.model.lr_scheduler_decay)
        elif self.conf.model.scheduler_type == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs // self.conf.model.t_max_ratio,
                eta_min=self.conf.model.eta_min
            )
        elif self.conf.model.scheduler_type == 'cosinelrscheduler':
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.conf.model.t_initial,
                lr_min=self.conf.model.lr_min,
                cycle_mul=self.conf.model.cycle_mul,
                cycle_decay=self.conf.model.cycle_decay,
                cycle_limit=self.conf.model.cycle_limit,
                warmup_t=self.conf.model.warmup_t,
                warmup_lr_init=self.conf.model.warmup_lr_init,
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def setup_metrics(self):
        metric_names = ['valid']
        if self.log_train:
            metric_names += ['train']
        if self.log_frame:
            metric_names += [f'valid_{t}' for t in range(self.sequence_length)]
        if self.log_train and self.log_frame:
            metric_names += [f'train_{t}' for t in range(self.sequence_length)]
        return {name: Metrics(self.num_classes, self.device) for name in metric_names}

    def get_resume_ckpt_path(self):
        ckpt_path = None
        if self.conf.trainer.resume_ckpt is not None:
            ckpt_path = self.conf.trainer.resume_ckpt
        elif self.conf.trainer.auto_resume:
            ckpt_path = get_latest_ckpt(self.conf.name)
        return ckpt_path

    def load_checkpoint(self, ckpt_path):
        if ckpt_path is None:
            return

        checkpoint = torch.load(ckpt_path, map_location=f'cuda:{self.device}')

        self.current_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_valid_miou = checkpoint['best_miou']
        self.current_step = checkpoint['global_step']
        (self.model.module if distributed() else self.model).load_state_dict(checkpoint['model_state_dict'])

        print_text(f'Loaded checkpoint {ckpt_path}')

    @func_rank_0
    def save_checkpoint(self, ckpt_path):
        checkpoint = {
            'epoch': self.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_valid_miou,
            'global_step': self.current_step,
            'model_state_dict': self.model.module.state_dict() if distributed() else self.model.state_dict(),
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)

    def fit(self):
        # initialize loggers
        if rank_0():
            wandb.init(
                project=self.conf.trainer.wandb_project,
                name=str(self.conf.name),
                config=OmegaConf.to_container(self.conf, resolve=True),
                mode='online' if self.conf.trainer.sync_wandb else 'offline'
            )

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.validate_epoch()
            self.save_checkpoint(Path(C.CKPT_ROOT) / str(self.conf.name) / C.CKPT_NAME_LAST)

        if rank_0():
            wandb.finish()

    def train_epoch(self):
        self.model.train()
        self.loss_history['train'] = {loss_name: 0. for loss_name in self.loss_names + ['loss']}

        if distributed():
            self.train_loader.sampler.set_epoch(self.current_epoch)

        epoch_start_time = time.time()
        pbar = tqdm(
            self.train_loader,
            desc=f'Train {self.current_epoch} / {self.num_epochs}',
            disable=not rank_0(),
            leave=False
        )
        for batch in pbar:
            # forward backward
            loss = self.forward_step(batch, name='train')
            self.backward_step(loss)

            lr = self.optimizer.param_groups[0]['lr']
            if self.current_step % self.log_frequency == 0:
                wlog({'lr': lr}, step=self.current_step)

            self.current_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        if self.conf.model.scheduler_type == 'cosinelrscheduler':
            self.scheduler.step(epoch=self.current_epoch + 1)
        else:
            self.scheduler.step()

        epoch_end_time = time.time()
        wlog({'epoch_duration': epoch_end_time - epoch_start_time}, step=self.current_step)
        wlog(
            {'sec_per_step': (epoch_end_time - epoch_start_time) / len(self.train_loader)},
            step=self.current_step
        )
        wlog({'epoch': self.current_epoch}, step=self.current_step)

        self.train_valid_epoch_end(name='train')

    def validate_epoch(self):
        self.model.eval()
        self.loss_history['valid'] = {loss_name: 0. for loss_name in self.loss_names + ['loss']}

        with torch.no_grad():
            pbar = tqdm(
                self.valid_loader,
                desc=f'Val {self.current_epoch} / {self.num_epochs}',
                disable=not rank_0(),
                leave=False
            )
            for batch in pbar:
                self.forward_step(batch, name='valid')

        valid_miou = self.metrics['valid'].get_metrics_dist()['miou']
        if self.best_valid_miou < valid_miou:
            self.save_checkpoint(
                Path(C.CKPT_ROOT) / self.conf.name / C.CKPT_NAME_RULE.format(self.current_epoch, valid_miou)
            )
            self.best_valid_miou = valid_miou
        self.train_valid_epoch_end(name='valid')

    def step(self, voxels):
        # forward pass
        model_out = self.model(voxels)

        # loss
        step_loss = self.compute_loss(voxels, model_out)
        step_loss['loss'] = sum([self.loss_weights[loss_name] * step_loss[loss_name] for loss_name in self.loss_names])

        return {
            'step_loss': step_loss,
            'pred': model_out['pred'],  # B, T, X, Y, Z, C
        }

    def compute_loss(self, gt, model_out):
        """
        model_out should contain 'pred' for ce and lovasz, 'mus' and 'logvars' for kl
        """
        step_loss = dict()
        for loss_name in self.loss_names:
            loss_func = self.loss_functions[loss_name]
            loss_params = None,
            if loss_name == 'ce':
                loss_params = model_out['pred'].permute(0, 5, 1, 2, 3, 4), gt
            elif loss_name == 'lovasz':
                loss_params = model_out['pred'].reshape(-1, self.num_classes), gt.reshape(-1, )
            elif loss_name == 'kl':
                loss_params = model_out['mus'], model_out['logvars']
            step_loss[loss_name] = loss_func(*loss_params)
        return step_loss

    def forward_step(self, batch, name='train'):
        voxels = batch['voxels'].to(self.device)

        # forward and loss
        with autocast('cuda', enabled=self.amp):
            step_out = self.step(voxels)

        # add to epoch level history
        step_loss = step_out['step_loss']
        for loss_name in step_loss.keys():
            self.loss_history[name][loss_name] = self.loss_history[name][loss_name] + step_loss[loss_name]

        if name == 'train' and self.current_step % self.log_frequency == 0:
            wlog({'step/loss': step_loss['loss']}, step=self.current_step)
            wlog({'step/cuda': torch.cuda.memory_reserved() / 2 ** 30}, step=self.current_step)

        # eval
        if self.log_train or name == 'valid':
            masks = batch['masks'].to(self.device) if self.use_valid_mask else None
            pred = step_out['pred']

            metrics_data = self.update_metrics(pred, voxels, mask=masks, name=name)

            if name == 'train' and self.current_step % self.log_frequency == 0:
                wlog({'step/IoU': metrics_data['bin_iou']}, step=self.current_step)
                wlog({'step/mIoU': metrics_data['miou']}, step=self.current_step)

            if self.log_frame:
                for t in range(self.sequence_length):
                    self.update_metrics(pred[:, t], voxels[:, t], mask=masks[:, t], name=f'{name}_{t}')

        return step_loss['loss']

    def backward_step(self, loss):
        self.optimizer.zero_grad()
        if self.amp:
            loss = self.scaler.scale(loss)
        loss.backward()

        if self.current_step >= self.conf.model.grad_clip_step and self.conf.model.grad_max_norm > 0:
            if self.amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.model.grad_max_norm)
            if self.current_step % self.log_frequency == 0:
                wlog({'grad_norm': grad_norm}, step=self.current_step)

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def train_valid_epoch_end(self, name='train'):
        # log loss
        loss_dict = self.loss_history[name]
        loss_len = len(self.train_loader) if name == 'train' else len(self.valid_loader)
        for loss_name in loss_dict.keys():
            wlog({f'{name}/{loss_name}': loss_dict[loss_name] / loss_len}, step=self.current_step)

        # log metrics
        if self.log_train or name == 'valid':
            self.log_metrics_and_reset(name)
            if self.log_frame:
                for t in range(self.sequence_length):
                    self.log_metrics_and_reset(f'{name}_{t}')

    def update_metrics(self, pred, gt, mask=None, name='train'):
        metrics = self.metrics[name]
        pred = get_pred_label(pred)
        if mask is not None:
            pred = pred[mask != 1]
            gt = gt[mask != 1]
        return metrics.update(pred.to(torch.int32), gt.to(torch.int32))

    def log_metrics_and_reset(self, name='train'):
        metrics = self.metrics[name]
        metrics_data = metrics.get_metrics_dist()
        metrics.reset()

        wlog({f'{name}/IoU': metrics_data['bin_iou']}, step=self.current_step)
        wlog({f'{name}/mIoU': metrics_data['miou']}, step=self.current_step)

        if self.log_class:
            for class_idx, class_name in enumerate(self.class_names):
                wlog({f'{name}/{class_name}': metrics_data['cls_iou'][class_idx]}, step=self.current_step)

    # inference
    def predict(self, save_rollout, save_voxel, save_layout):
        self.model.eval()
        with torch.no_grad():
            self.predict_epoch('train', save_rollout, save_voxel, save_layout)
            self.predict_epoch('valid', save_rollout, save_voxel, save_layout)

    def predict_epoch(self, name, save_rollout, save_voxel, save_layout):
        loader = self.train_loader if name == 'train' else self.valid_loader
        assert loader.batch_size == 1, 'inference batch size must be 1'

        if name == 'train' and not save_rollout and save_voxel is None and save_layout is None:
            return

        pbar = tqdm(loader, desc=f'{name}', disable=not rank_0(), leave=False)
        for i, batch in enumerate(pbar):
            batch_out = self.predict_batch(name, batch)

            if save_rollout:
                rollout_root = Path(C.ROLLOUT_NAME) / self.conf.dataset.dataset / self.conf.name
                self.save_batch_rollout(batch_out['hexplane'], batch_out['path'], rollout_root)

            if save_voxel is not None:
                rank = 0 if rank_0() else dist.get_rank()
                voxel_root = (Path(C.OUTPUT_ROOT) / C.VOXEL_NAME / self.conf.name /
                              f'{name}-{i + rank * len(loader)}')
                voxel_root.mkdir(parents=True, exist_ok=True)
                self.save_batch_voxel(batch_out['pred'], batch_out['voxels'], voxel_root)

            if save_layout is not None:
                layout_root = Path(C.LAYOUT_NAME) / self.conf.dataset.dataset / str(self.sequence_length)
                self.save_batch_layout(batch_out['voxels'], batch_out['path'], layout_root, save_layout)

    def predict_batch(self, name, batch):
        voxels = batch['voxels'].to(self.device)
        paths = batch['paths']

        with autocast('cuda', enabled=self.amp):
            batch_out = self.model(voxels)

        if name == 'valid':
            masks = batch['masks'].to(self.device) if self.use_valid_mask else None
            pred = batch_out['pred']

            self.update_metrics(pred, voxels, mask=masks, name=name)

            if self.log_frame:
                for t in range(self.sequence_length):
                    self.update_metrics(pred[:, t], voxels[:, t], mask=masks[:, t], name=f'{name}_{t}')

        return {
            'voxels': voxels,
            'pred': batch_out['pred'],
            'hexplane': batch_out['hexplane'],
            'path': paths[0][0]
        }

    def save_batch_rollout(self, hexplane, path, root):
        save_path = root / parse_rollout_path(self.conf.dataset, path)  # rollout/carlasc/0000_exp/Train/Town...
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, rollout=hexplane_to_rollout(hexplane).cpu().numpy())

    def save_batch_voxel(self, pred, voxels, folder):
        pred = get_pred_label(pred).cpu().numpy().astype(np.uint8)
        voxels = voxels.cpu().numpy().astype(np.uint8)
        for t in range(self.sequence_length):
            pred[:, t].tofile(folder / f'{t}_pred.npy')
            voxels[:, t].tofile(folder / f'{t}_orig.npy')

    def save_batch_layout(self, voxels, path, layout_folder, down_size):
        save_path = layout_folder / parse_rollout_path(self.conf.dataset, path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        layout = convert_voxels_to_layouts(voxels.squeeze(0), down_size).cpu().numpy().astype(np.uint8)
        np.savez_compressed(save_path, layout=layout)

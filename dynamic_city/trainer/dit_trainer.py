import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import dynamic_city.utils.constants as C
from dynamic_city.dataset.builder import get_hexplane_dataloaders
from dynamic_city.diffusion import create_diffusion
from dynamic_city.diffusion.models import DiT_models
from dynamic_city.utils.ckpt_utils import get_latest_ckpt, load_conf
from dynamic_city.utils.dist_utils import distributed, flush_text, func_rank_0, print_text, rank_0, wlog, write_text
from dynamic_city.utils.dit_utils import requires_grad, update_ema
from dynamic_city.utils.hexplane_utils import get_rollout_mask


class DiTTrainer:
    def __init__(self, conf, device, load_data=True):
        self.conf = conf
        self.device = device

        # vae conf
        self.vae_conf = load_conf(conf.vae_name)

        # dataset conf
        self.t = self.vae_conf.dataset.sequence_length
        self.x, self.y, self.z = self.vae_conf.dataset.spatial_size
        self.hex_t = self.t // (2 ** self.vae_conf.model.hex_down_t)
        self.hex_x = self.x // (2 ** self.vae_conf.model.hex_down_x) // (2 ** self.vae_conf.model.down_x)
        self.hex_y = self.y // (2 ** self.vae_conf.model.hex_down_y) // (2 ** self.vae_conf.model.down_y)
        self.hex_z = self.z // (2 ** self.vae_conf.model.hex_down_z) // (2 ** self.vae_conf.model.down_z)
        self.hex_txyz = self.hex_t, self.hex_x, self.hex_y, self.hex_z

        # trainer conf
        self.num_epochs = conf.trainer.num_epochs
        self.log_frequency = conf.trainer.log_frequency
        self.ckpt_frequency = conf.trainer.ckpt_frequency

        self.train_dataloader = None
        if load_data:
            self.train_dataloader = get_hexplane_dataloaders(self.conf, self.vae_conf)
            print_text(f'Dataset Size: {len(self.train_dataloader.dataset)}')

        # pro valid mask used to calculate loss
        self.rollout_mask = get_rollout_mask(self.hex_txyz).to(self.device)  # image_size, image_size
        self.image_size = self.rollout_mask.shape[-1]

        # patch mask used for faster forward
        patch_mask_size = (self.hex_t // 2, self.hex_x // 2, self.hex_y // 2, self.hex_z // 2)
        self.patch_mask = get_rollout_mask(patch_mask_size)
        self.patch_mask = self.patch_mask.to(self.device).half()
        self.patch_mask = self.patch_mask.flatten(2).transpose(1, 2).expand(conf.dataset.batch_size, -1, 1)

        # setup model and diffusion
        self.model, self.ema = self.setup_model()
        self.diffusion = create_diffusion(timestep_respacing="")
        print_text(f'DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        # setup optimizer
        self.optimizer = self.setup_optimizer()
        assert self.conf.trainer.amp
        self.scaler = GradScaler('cuda')

        # counters
        self.current_step = 0
        self.log_steps = 0
        self.running_loss = 0
        self.current_epoch = 0
        self.start_time = 0

        # resume
        self.load_checkpoint(self.get_resume_ckpt_path())

    def setup_model(self):
        model = DiT_models[self.conf.model.model](
            in_channels=self.vae_conf.model.latent_channels,
            x_attn=self.conf.model.x_attn,
            txyz=self.hex_txyz,
            seq_len=self.t,
            patch_mask=self.patch_mask,
            hex_cond=self.conf.model.hex_cond,
            layout_cond=self.conf.model.layout_cond,
            traj_cond=self.conf.model.traj_cond,
            cmd_cond=self.conf.model.cmd_cond,
            traj_size=3 * self.t,
        ).to(self.device)
        ema = deepcopy(model).to(self.device)
        requires_grad(ema, False)

        # optional ddp
        if distributed():
            model = DDP(model, device_ids=[self.device])

        update_ema(ema, (model.module if distributed() else model), decay=0)
        model.train()
        ema.eval()
        return model, ema

    def setup_optimizer(self):
        return AdamW(
            self.model.parameters(),
            lr=self.conf.model.learning_rate,
            weight_decay=self.conf.model.weight_decay
        )

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
        self.current_step = checkpoint['current_step']
        (self.model.module if distributed() else self.model).load_state_dict(checkpoint['model_state_dict'])
        self.ema.load_state_dict(checkpoint['ema'])

        flush_text()
        print_text(f'Loaded checkpoint {ckpt_path}')

    @func_rank_0
    def save_checkpoint(self, ckpt_path):
        checkpoint = {
            'epoch': self.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'model_state_dict': self.model.module.state_dict() if distributed() else self.model.state_dict(),
            'ema': self.ema.state_dict(),
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)

    def forward_step(self, batch):
        x = batch['hexplane'].to(self.device)
        hex_cond = batch['hex_cond'].to(self.device)
        layout_cond = batch['layout_cond'].to(self.device).to(x.dtype)
        traj_cond = batch['traj_cond'].to(self.device).to(x.dtype)
        cmd_cond = batch['cmd_cond'].to(self.device)

        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        model_kwargs = {
            'hexplane': hex_cond,
            'layout': layout_cond,
            'traj': traj_cond,
            'cmd': cmd_cond,
        }

        with autocast('cuda'):
            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs, mask=self.rollout_mask)

        loss = loss_dict['loss'].mean()
        return loss

    def backward_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        update_ema(self.ema, (self.model.module if distributed() else self.model))
        self.optimizer.zero_grad()

    def after_step(self, loss):
        if self.current_step > 0:
            flush_text()

        self.running_loss += loss.item()
        self.log_steps += 1
        self.current_step += 1

        if self.current_step % self.log_frequency == 0:
            # avg loss
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=self.device)
            if distributed():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

            # time
            end_time = time.time()
            steps_ps = self.log_steps / (end_time - self.start_time)

            print_text(f"(step={self.current_step:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_ps:.2f}")
            if rank_0():
                wlog({'train_loss': avg_loss, 'train_steps_per_sec': steps_ps}, step=self.current_step)
                wlog({'cuda': torch.cuda.memory_reserved() / 2 ** 30}, step=self.current_step)

            # reset
            self.running_loss = 0
            self.log_steps = 0
            self.start_time = time.time()

        step_text = f'Step: {self.current_step} / {len(self.train_dataloader) * self.num_epochs} | Loss: {loss.item():.4f}'
        write_text(step_text)

        # save DiT checkpoint:
        if self.current_step % self.ckpt_frequency == 0 and self.current_step > 0:
            checkpoint_path = Path(C.CKPT_ROOT) / self.conf.name / f'{self.current_step:07d}.ckpt'
            self.save_checkpoint(checkpoint_path)
            last_path = Path(C.CKPT_ROOT) / self.conf.name / C.CKPT_NAME_LAST
            self.save_checkpoint(last_path)
            flush_text()
            print_text(f'Saved checkpoint {checkpoint_path}')
            if distributed():
                dist.barrier()

    def train_epoch(self):
        self.model.train()
        if distributed():
            self.train_dataloader.sampler.set_epoch(self.current_epoch)
        epoch_start_time = time.time()
        flush_text()
        print_text(f'Starting training epoch [ {self.current_epoch} / {self.num_epochs} ]')
        for batch in self.train_dataloader:
            loss = self.forward_step(batch)
            self.backward_step(loss)
            self.after_step(loss)
        epoch_end_time = time.time()
        wlog({'epoch_duration': epoch_end_time - epoch_start_time}, step=self.current_step)
        wlog({'epoch': self.current_epoch}, step=self.current_step)

    def fit(self):
        # initialize loggers
        if rank_0():
            wandb.init(
                project=self.conf.trainer.wandb_project,
                name=str(self.conf.name),
                config=OmegaConf.to_container(self.conf, resolve=True),
                mode='online' if self.conf.trainer.sync_wandb else 'offline'
            )

        print_text(f'Training for {self.num_epochs} epochs')
        self.start_time = time.time()
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()

        if rank_0():
            wandb.finish()

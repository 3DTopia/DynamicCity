"""
Inference script for saving DiT results.
Example: python infer_dit.py -d DynamicCityDiT --best_vae
"""

import argparse
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from tqdm import tqdm

import dynamic_city.utils.constants as C
from dynamic_city.diffusion import create_diffusion
from dynamic_city.trainer.dit_trainer import DiTTrainer
from dynamic_city.trainer.vae_trainer import VAETrainer
from dynamic_city.utils.ckpt_utils import get_dit_ckpt, get_vae_ckpt, load_conf
from dynamic_city.utils.dist_utils import distributed, print_text, rank_0
from dynamic_city.utils.hexplane_utils import rollout_to_hexplane
from dynamic_city.utils.torch_utils import cleanup_dist, set_seed, set_tf32, setup_dist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dit_name', type=str, required=True)
    parser.add_argument('-n', '--num_samples', type=int, default=16)

    # conditions
    parser.add_argument('-l', '--hexplane', type=int, default=1)
    parser.add_argument('--layout', type=Path, default=None)
    parser.add_argument('--command', type=int, default=None)
    parser.add_argument('--trajectory', type=Path, default=None)

    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)
    parser.add_argument('--best_vae', action='store_true')

    parser.add_argument('--cfg_scale', type=float, default=4.0)
    parser.add_argument('--num_sampling_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    warnings.filterwarnings('ignore')

    # load vae and dit configs
    args = get_args()

    # load conf
    dit_conf = load_conf(args.dit_name)
    OmegaConf.set_struct(dit_conf, False)
    dit_conf.trainer.auto_resume = True
    if args.ckpt_step > 0:
        dit_conf.trainer.resume_ckpt = get_dit_ckpt(dit_conf.name, args.ckpt_step)[0]

    vae_conf = load_conf(dit_conf.vae_name)
    OmegaConf.set_struct(vae_conf, False)
    vae_conf.trainer.auto_resume = True
    if args.best_vae:
        vae_conf.trainer.resume_ckpt = get_vae_ckpt(vae_conf.name)[0]

    dit_conf.dataset.batch_size = 2  # classifier free guidance

    # torch, ddp, and seed setup
    torch.set_grad_enabled(False)
    set_tf32(dit_conf.trainer.tf32)
    rank, device = setup_dist()
    set_seed(args.seed, dit_conf.trainer.deterministic, rank)

    # build trainer
    vae_trainer = VAETrainer(vae_conf, device)
    dit_trainer = DiTTrainer(dit_conf, device, load_data=False)

    vae_model = vae_trainer.model
    dit_model = dit_trainer.ema.cuda().eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # setup for inference
    channels = vae_conf.model.get('latent_channels', 0)
    image_size = dit_trainer.image_size

    if args.trajectory is not None:
        trajectories = np.load(args.trajectory)
        args.num_samples = trajectories.shape[0]
        args.hexplane = 1
        trajectories = rearrange(trajectories, 'b t c -> b 1 t c')
    else:
        trajectories = np.random.randn(args.num_samples, args.hexplane, vae_conf.dataset.sequence_length, 3)

    if args.layout is not None:
        layouts = list(args.layout.glob('**/*.npz'))

    save_path = Path(C.OUTPUT_ROOT) / C.GEN_NAME / f'{args.dit_name}{args.postfix}'

    start = time.time()

    world_size = dist.get_world_size() if distributed() else 1
    for sample_index in tqdm(range(rank, args.num_samples, world_size), disable=not rank_0()):
        hex_cond = torch.zeros(1, channels, image_size, image_size, device=device)

        for sequence_index in range(args.hexplane):
            noise = torch.randn(1, channels, image_size, image_size, device=device)
            noise = torch.cat([noise, noise], 0)

            hex_cond = torch.cat([hex_cond, hex_cond], 0)

            traj_cond = torch.tensor(trajectories[sample_index, sequence_index], device=device).half().unsqueeze(0)
            traj_cond = torch.cat([traj_cond, traj_cond], 0)

            if args.command is not None:
                command = args.command
                if command == -1:
                    command = sample_index % 4
            else:
                command = 0
            cmd_cond = torch.tensor([command], device=device)
            cmd_cond = torch.cat([cmd_cond, cmd_cond], 0)

            if args.layout is not None:
                layout = np.load(random.choice(layouts))['layout']
            else:
                layout = np.random.randn(
                    vae_conf.dataset.sequence_length,
                    vae_conf.dataset.spatial_size[0] // 4, vae_conf.dataset.spatial_size[1] // 4
                )
            layout_cond = torch.tensor(layout, device=device).half().unsqueeze(0)
            layout_cond = torch.cat([layout_cond, layout_cond], 0)

            model_kwargs = dict(
                cfg_scale=args.cfg_scale,
                hexplane=hex_cond,
                layout=layout_cond,
                cmd=cmd_cond,
                traj=traj_cond,
                inference=True,
                drop_hex=sequence_index == 0,
                drop_layout=args.layout is None,
                drop_traj=args.trajectory is None,
                drop_cmd=args.command is None,
            )

            with torch.no_grad():
                with autocast():
                    samples = diffusion.p_sample_loop(
                        dit_model.forward_with_cfg, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                        device=device
                    )

                    samples, _ = samples.chunk(2, dim=0)
                    hex_cond = samples

                    samples = rollout_to_hexplane(samples, dit_trainer.hex_txyz)
                    model_output = (vae_model.module if distributed() else vae_model).decoder(samples)

                    voxel = torch.softmax(model_output.to(torch.float32), dim=-1)
                    voxel = voxel.argmax(dim=-1)

            voxel = voxel.cpu().numpy().astype(np.uint8)

            if voxel.sum() == 0:
                print(f'Sample {sample_index} failed on sequence {sequence_index}.')
                break

            save_folder = Path(save_path) / f'{sample_index}'
            save_folder.mkdir(parents=True, exist_ok=True)

            for t in range(voxel.shape[1]):
                voxel[:, t].tofile(save_folder / f'{sequence_index * vae_conf.dataset.sequence_length + t}.npy')

    print_text(f"Time: {time.time() - start}")
    cleanup_dist()


if __name__ == '__main__':
    main()
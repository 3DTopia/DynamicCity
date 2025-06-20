from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import dynamic_city.utils.constants as C
from dynamic_city.utils.data_utils import get_command


class HexPlaneDataset(ABC, Dataset):
    def __init__(self, dit_conf, vae_conf, split):
        self.dit_conf = dit_conf
        self.vae_conf = vae_conf
        self.split = split

        self.data_folder = Path('rollout') / vae_conf.dataset.dataset / vae_conf.name / vae_conf.dataset.split[split]
        self.vae_name = vae_conf.name
        self.sequence_length = vae_conf.dataset.sequence_length

        self.hex_cond = dit_conf.model.hex_cond
        self.layout_cond = dit_conf.model.layout_cond
        self.traj_cond = dit_conf.model.traj_cond
        self.cmd_cond = dit_conf.model.cmd_cond

        self.hexplanes, self.conditions, self.trajectories, self.turns = self.prepare()

    def __len__(self):
        return len(self.hexplanes)

    def __getitem__(self, index):
        hexplane = np.load(self.hexplanes[index])['rollout'].squeeze()

        if self.hex_cond:
            hex_cond = np.load(self.conditions[index])['rollout'].squeeze()
        else:
            hex_cond = 0

        if self.cmd_cond:
            command = get_command(
                self.trajectories[index], self.dit_conf.dataset.angle_thr_mul,
                self.dit_conf.dataset.forward_thr_mul
            ).value
        else:
            command = 0

        if self.layout_cond:
            layout = np.load(
                str(self.hexplanes[index])
                .replace(C.ROLLOUT_NAME, C.LAYOUT_NAME)
                .replace(self.vae_name, str(self.sequence_length))
            )['layout']
        else:
            layout = 0

        return {
            'hexplane': hexplane,
            'hex_cond': hex_cond,
            'layout_cond': layout,
            'traj_cond': self.trajectories[index],
            'cmd_cond': command,
            'path': str(self.hexplanes[index]),
            'turn': self.turns[index],
        }

    @abstractmethod
    def prepare(self):
        pass

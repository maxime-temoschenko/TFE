
import h5py
import json
import math

import random
import torch

import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from typing import *
from datetime import datetime


# Adapted from https://github.com/francois-rozet/sda/blob/qg/sda/utils.py#L58
class SequenceDataset(Dataset):
    def __init__(self,
                 file: Path,
                 window : int = None,
                 flatten: bool = False):
        super().__init__()
        with h5py.File(file, mode='r') as f:
            self.data = f['data'][:]
            self.date = f['date'][:]
        self.date = [datetime.fromisoformat(d.decode()[:-6]) for d in self.date]
        self.window = window
        self.flatten = flatten

        # Spatial Encoding
        _, _, self.y, self.x = self.data.shape

        grid = torch.stack(
            torch.meshgrid(
                2*torch.pi*torch.arange(self.y)/self.y,
                2*torch.pi*torch.arange(self.x)/self.x,
                indexing='ij'
            )

        )
        self.spatial_encoding = torch.cat((grid.cos(), grid.sin()), dim=0)
        print(f"Spatial Encoding Shape : {self.spatial_encoding.shape}")

    def __len__(self) -> int:
        return len(self.data) - self.window + 1
    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        data_x = torch.from_numpy(self.data)

        i = torch.randint(0, len(self.data) - self.window + 1, size=()).item()

        traj_x = torch.narrow(data_x, dim=0, start=i, length=self.window)
        traj_date = self.date[i: i + self.window]

        # Embedding :
        date_enc = torch.tensor([
            date_encoding(d.timetuple().tm_yday, d.hour) for d in traj_date
        ]).float()
        date_enc = date_enc.view(self.window*4,1,1).expand(-1,self.y, self.x)# [WINDOW*4, Y,X]

        context = torch.cat([date_enc, self.spatial_encoding], dim=0)

        if self.flatten:
            return traj_x.flatten(0, 1), { 'context' : context}

        return traj_x, { 'context' : context}
#
class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Path,
        window: int = None,
        flatten: bool = False,
    ):
        super().__init__()

        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]

        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i])

        if self.window is not None:
            i = torch.randint(0, len(x) - self.window + 1, size=())
            x = torch.narrow(x, dim=0, start=i, length=self.window)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}

def date_encoding(day : int, hour : int):
    day_emb = (np.sin(2*np.pi*day/365), np.cos(2*np.pi*day/365))
    hour_emb = (np.sin(2*np.pi*hour/365), np.cos(2*np.pi*hour/365))
    return day_emb+hour_emb
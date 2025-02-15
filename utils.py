
import h5py
import json
import math

import preprocess
import matplotlib.pyplot as plt
import random
import torch

import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from typing import *
from datetime import datetime

import wandb



# Adapted from https://github.com/francois-rozet/sda/blob/qg/sda/utils.py#L58
class SequenceDataset(Dataset):
    def __init__(self,
                 file: Path,
                 window : int = None,
                 flatten: bool = False):
        super().__init__()
        with h5py.File(file, mode='r') as f:
            # TODO : Remove 12, these are for debugging purpose
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

        #context = torch.cat([date_enc, self.spatial_encoding], dim=0)
        context = self.spatial_encoding
        if self.flatten:
            return traj_x.flatten(0, 1), { 'context' : context}

        return traj_x, { 'context' : context}

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

# TODO : Generalize for infos about variables, clean code
'''
def plot_sample(batch, info, mask, path_unnorm, samples, step=4):
    B, Z, Y, X = batch.shape
    redim_batch = batch.view(B, info['window'], info['channels'], Y, X).permute(0, 2, 1, 3, 4)
    print(redim_batch.shape)
    for i in range(info['channels']):
        redim_batch[:, i, ...] = preprocess.unnormalize_ds(redim_batch[:, i, ...], info['var_index'][i],
                                                           normfile_path=path_unnorm,
                                                           normalization_mode='zscore')
    traj = torch.where(mask.unsqueeze(0).unsqueeze(0).bool(), redim_batch,
                       torch.tensor(float('nan'), dtype=redim_batch.dtype))
    data = traj

    s, variables, timesteps, y_dim, x_dim = data.shape
    selected_timesteps = range(0, timesteps, step)
    fig, axes = plt.subplots(nrows=info['channels'] * samples, ncols=len(selected_timesteps), figsize=(15, 5 * samples))
    cbar_axes = []
    for s in range(samples):
        for var in range(variables):
            mean, std = np.nanmean(data[s, var, :]), np.nanstd(data[s, var, :])
            print(f"{info['var_index'][var]}  Mean : {mean}, Var: {std}")
            for i, t in enumerate(selected_timesteps):
                vmin, vmax = np.nanmin(data[s, var, t]), np.nanmax(data[s, var, t])
                ax = axes[info['channels'] * s + var, i]  # Correct row indexing
                img_data = data[s, var, t].numpy()
                img_data = np.ma.masked_invalid(img_data)
                cmap = plt.get_cmap('viridis' if var == 0 else 'plasma')
                cmap.set_bad(color='black')
                img = ax.imshow(img_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if t == 0:
                    list_var = ["Temperature[°C]", "Wind Speed[M/S]"]
                    ax.set_ylabel(f'{list_var[var]}')
                if var == 0:
                    ax.set_title(f"{t} Hours")
                cbar_axes.append(fig.colorbar(img, ax=ax))

    for s in range(samples):
        fig.text(
            -0.05, 1 - (2 * s + 1) / (2 * samples),
            f"Sample {s}",
            va='center', ha='center', rotation=90, fontsize=12, fontweight='bold'
        )
    fig.suptitle("Visualization of samples", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig
'''

def plot_sample(batch,info,mask, samples, step=4, unnormalize=True, path_unnorm = 'data/norm_params.h5'):
    print(batch.shape)
    B, Z, Y , X = batch.shape
    redim_batch = batch.view(B, info['window'], info['channels'], Y,X).permute(0, 2, 1, 3, 4)
    print(redim_batch.shape)
    if unnormalize == True:
        for i in range(info['channels']):
            redim_batch[:,i, ...] = preprocess.unnormalize_ds(redim_batch[:,i, ...], info['var_index'][i], normfile_path=path_unnorm ,normalization_mode='zscore')
    traj= torch.where(mask.unsqueeze(0).unsqueeze(0).bool(), redim_batch, torch.tensor(float('nan'), dtype=redim_batch.dtype))
    data = traj  # Expected shape: [samples, timesteps, variables, y, x]
    print(traj.shape)

    print("Input shape:", data.shape)
    s, variables, timesteps, y_dim, x_dim = data.shape
    selected_timesteps = range(0, timesteps, step)
    fig, axes = plt.subplots(nrows=info['channels'] * samples, ncols=len(selected_timesteps), figsize=(15, 5 * samples))
    cbar_axes = []
    for s in range(samples):
        for var in range(variables):
            mean, std =  np.nanmean(data[s,var,:]), np.nanstd(data[s,var,:])
            print(f"{info['var_index'][var]}  Mean : {mean}, Var: {std}")
            for i, t in enumerate(selected_timesteps):
                vmin, vmax = np.nanmin(data[s, var, t]), np.nanmax(data[s, var, t])
                ax = axes[info['channels'] * s + var, i]  # Correct row indexing
                img_data = data[s, var,t].numpy()
                img_data = np.ma.masked_invalid(img_data)
                cmap = plt.get_cmap('viridis' if var == 0 else 'plasma')
                cmap.set_bad(color='black')
                img = ax.imshow(img_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if t == 0:
                    list_var = ["Temperature[°C]", "Wind Speed[M/S]"]
                    ax.set_ylabel(f'{list_var[var]}')
                if var == 0:
                    ax.set_title(f"{t} Hours")
                cbar_axes.append(fig.colorbar(img, ax=ax))

    for s in range(samples):
        fig.text(
            -0.05, 1 - (2 * s + 1) / (2 * samples),
            f"Sample {s}",
            va='center', ha='center', rotation=90, fontsize=12, fontweight='bold'
        )
    fig.suptitle("Visualization of samples", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig

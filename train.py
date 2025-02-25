import sys
sys.path.append('.')
import os
import h5py
import math
import torch
import numpy as np
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from pathlib import Path
from utils import SequenceDataset, plot_sample, DateEmbedding
from score import ScoreUNet, MCScoreWrapper, VPSDE
from score import VPSDE

def constructEmbedding(date_embedding, dic):
    return torch.concat((dic['context'], date_embedding(dic['frac_day_of_year'], dic['frac_hour_of_day']).unsqueeze(1)), dim=1)


PATH_DATA = Path('./data/processed')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

window=12

with h5py.File(PATH_DATA / "mask.h5", "r") as f:
    mask = torch.tensor(f["dataset"][:], dtype=torch.float32, device=device).unsqueeze(0)
    mask_cpu = mask.detach().clone().cpu()

if torch.isnan(mask).any():
    raise ValueError("Mask contains NaN values!")
#Date Embedding
date_embedding  = DateEmbedding()
# Load Dataset
trainset = SequenceDataset(PATH_DATA / "train.h5", window=window, flatten=True)
validset = SequenceDataset(PATH_DATA / "test.h5", window=window, flatten=True)

# Dimensions
channels, y_dim, x_dim = trainset[0][0].shape #channels = (#var_keeps+1) * window
# CONFIG
TRAIN_CONFIG = {
    "epochs": 10000,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "embedding": 64,
    "activation": "SiLU",
    "eta": 5e-3,
}
MODEL_CONFIG = { 'hidden_channels' : [64],
'hidden_blocks' : [2],
'spatial' : 2,
'channels' : channels,
'context' : 5,
'embedding' : 64 }
CONFIG = {**TRAIN_CONFIG, **MODEL_CONFIG}
run = wandb.init(
    project="Denoiser-Training",
    config= CONFIG
)
'''
Definition of Denoiser and Scheduler
'''

# Denoiser and Scheduler
score_unet = ScoreUNet(**MODEL_CONFIG)
vpsde = VPSDE(score_unet, shape=(channels, y_dim, x_dim), eta=TRAIN_CONFIG['eta']).cuda()



## TRAINING LOOP
trainloader = DataLoader(trainset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=1, persistent_workers=True)
validloader = DataLoader(validset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, num_workers=1, persistent_workers=True)


optimizer = optim.AdamW([*vpsde.parameters(), *date_embedding.parameters()], lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])

# Define Learning Rate Scheduler
if TRAIN_CONFIG["scheduler"] == "linear":
    lr_lambda = lambda t: 1 - (t / TRAIN_CONFIG["epochs"])
elif TRAIN_CONFIG["scheduler"] == "cosine":
    lr_lambda = lambda t: (1 + math.cos(math.pi * t / TRAIN_CONFIG["epochs"])) / 2
elif TRAIN_CONFIG["scheduler"] == "exponential":
    lr_lambda = lambda t: math.exp(-7 * (t / TRAIN_CONFIG["epochs"]) ** 2)
else:
    raise ValueError("Invalid scheduler type")

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

checkpoint_dir = "checkpoints/multi_attention_config_spatial_T2m_U10m_2000_2014"
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0


all_losses_train = []
all_losses_valid = []
for epoch in (bar := trange(start_epoch, TRAIN_CONFIG["epochs"], ncols=88)):
    losses_train = []
    losses_valid = []

    # Training
    vpsde.train()
    for i, (batch, dic) in enumerate(trainloader):
        batch = batch.to(device)
        c = constructEmbedding(date_embedding,dic).to(device)
        if torch.isnan(batch).any():
            raise ValueError("batch contains NaN values!")
        batch = batch.to(device)
        optimizer.zero_grad()

        # Mask
        mask_batch = mask.to(device).expand_as(batch)
        w = mask_batch.float()


        loss = vpsde.loss(batch, w=w, c=c)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.detach())

    # Evaluation
    vpsde.eval()
    with torch.no_grad():
        for batch, dic in validloader:
            batch = batch.to(device)
            c = constructEmbedding(date_embedding,dic).to(device)
            mask_batch = mask.to(device).expand_as(batch)
            w = mask_batch.float()

            loss = vpsde.loss(batch, w=w, c=c)
            losses_valid.append(loss.detach())

    loss_train = torch.stack(losses_train).mean().item()
    loss_valid = torch.stack(losses_valid).mean().item()
    print(f"Train Loss : {loss_train}, Valid Loss : {loss_valid}")
    lr = optimizer.param_groups[0]['lr']
    log = {"Train Loss" : loss_train, "Valid Loss" : loss_valid, "lr" : lr}
    all_losses_train.append(loss_train)
    all_losses_valid.append(loss_valid)

    #Save model sometimes
    if epoch % 10 == 0  :
        checkpoint_path = os.path.join(checkpoint_dir, f"multi_attention_config_spatial_T2m_U10m_2000_2014_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': vpsde.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
        }, checkpoint_path)
        print(f"Model saved at {checkpoint_path}")
        # Log some plots
        with torch.no_grad():
            myLoader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1,
                                    persistent_workers=True)
            batch, dic = next(iter(myLoader))
            c = constructEmbedding(date_embedding, dic).to(device)
            sampled_traj = vpsde.sample(mask,c=c,shape=(10,), steps=128, corrections=1).detach().cpu()
            batch = batch[0]
            x = batch.repeat((3,) + (1,) * len(batch.shape))
            t = torch.rand(x.shape[0], dtype=x.dtype)
            c = constructEmbedding(date_embedding, dic)[0]
            c = c.repeat((3,) + (1,) * len(c.shape))
            # Noise  Levels to plot
            t[0] = 0.5
            t[1] = 0.9
            t[2] = 1
            t = t.to(device)
            x = x.to(device)
            c = c.to(device)
            x_t = vpsde.forward(x, t, train=False)
            print(f"x_t = {x_t.shape}, t : {t.shape}, c : {c.shape}, x : {x.shape}")
            x_0 = vpsde.denoise(x_t, t, c).detach().cpu()
            x_t = x_t.detach().cpu()
            x = x.detach().cpu()
        path_unnorm = PATH_DATA / "train.h5"
        info = {'var_index': ['T2m', 'U10m'], 'channels': 2, 'window': 12}
        fig = plot_sample(sampled_traj, info, mask_cpu,  samples=5, step=3, unnormalize=True, path_unnorm = path_unnorm)
        log['samples'] = wandb.Image(fig)


        new_tensor = torch.stack((x, x_t, x_0), dim=1).flatten(0, 1)
        fig = plot_sample(new_tensor, info, mask_cpu, samples=9, step=3, unnormalize=False, path_unnorm = path_unnorm)
        log['chart'] =  wandb.Image(fig)
    scheduler.step()
    wandb.log(log)

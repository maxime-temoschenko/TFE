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
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange
from pathlib import Path
from utils import SequenceDataset, plot_sample, DateEmbedding
from score import ScoreUNet, MCScoreWrapper, VPSDE

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()


def constructEmbedding(date_embedding, dic, device):
    return torch.concat(
        (dic['context'].to(device), date_embedding(dic['frac_day_of_year'].to(device), dic['frac_hour_of_day'].to(device)).unsqueeze(1)),
        dim=1
    )


def main():
    args = parse_args()
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"Using device: {device} on rank {local_rank}")

    PATH_DATA = Path('./data/processed')

    # Load mask
    with h5py.File(PATH_DATA / "mask.h5", "r") as f:
        mask = torch.tensor(f["dataset"][:], dtype=torch.float32, device=device).unsqueeze(0)
        mask_cpu = mask.detach().clone().cpu()

    if torch.isnan(mask).any():
        raise ValueError("Mask contains NaN values!")
    window = 12
    # Date Embedding
    date_embedding = DateEmbedding().to(device)

    # Load Dataset
    trainset = SequenceDataset(PATH_DATA / "train.h5", window=window, flatten=True)
    validset = SequenceDataset(PATH_DATA / "test.h5", window=window, flatten=True)

    # Distributed Sampler
    train_sampler = DistributedSampler(trainset)
    valid_sampler = DistributedSampler(validset, shuffle=False)

    # Dimensions
    channels, y_dim, x_dim = trainset[0][0].shape

    # CONFIG
    TRAIN_CONFIG = {
        "window" : 12,
        "epochs": 10000,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "embedding": 64,
        "activation": "SiLU",
        "eta": 5e-3,
    }
    MODEL_CONFIG = {
        'hidden_channels': [64, 128, 256, 512, 1024],
        'attention_levels': [4],
        'hidden_blocks': [2, 3, 3, 3,3],
        'spatial': 2,
        'channels': 24,
        'context': 5,
        'embedding': 64
    }
    CONFIG = {**TRAIN_CONFIG, **MODEL_CONFIG}

    # Initialize wandb only on rank 0
    if local_rank == 0:
        run = wandb.init(
            project="Denoiser-Training",
            config=CONFIG
        )
        PATH = Path('.')
        PATH_SAVE = PATH / f'checkpoints/{run.name}_{run.id}'
        PATH_SAVE.mkdir(parents=True, exist_ok=True)


    # Denoiser and Scheduler
    score_unet = ScoreUNet(**MODEL_CONFIG).to(device)
    vpsde = VPSDE(score_unet, shape=(channels, y_dim, x_dim), eta=TRAIN_CONFIG['eta']).to(device)

    # Wrap models with DDP
    vpsde = nn.parallel.DistributedDataParallel(vpsde, device_ids=[local_rank])
    date_embedding = nn.parallel.DistributedDataParallel(date_embedding, device_ids=[local_rank])

    # Dataloaders
    trainloader = DataLoader(
        trainset,
        batch_size=TRAIN_CONFIG["batch_size"] // dist.get_world_size(),
        sampler=train_sampler,
        num_workers=1,
        persistent_workers=True
    )
    validloader = DataLoader(
        validset,
        batch_size=TRAIN_CONFIG["batch_size"] // dist.get_world_size(),
        sampler=valid_sampler,
        num_workers=1,
        persistent_workers=True
    )

    optimizer = optim.AdamW(
        [*vpsde.parameters(), *date_embedding.parameters()],
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

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


    start_epoch = 0
    for epoch in  (bar := trange(start_epoch, TRAIN_CONFIG["epochs"], ncols=88)):
        train_sampler.set_epoch(epoch)
        vpsde.train()
        losses_train = []

        for batch, dic in trainloader:
            batch = batch.to(device)
            c = constructEmbedding(date_embedding, dic, device).to(device)

            if torch.isnan(batch).any():
                raise ValueError("Batch contains NaN values!")

            optimizer.zero_grad()
            mask_batch = mask.to(device).expand_as(batch)
            w = mask_batch.float()
            loss = vpsde.module.loss(batch, w=w, c=c)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.detach())

        loss_train = torch.stack(losses_train).mean().item()
        loss_train_tensor = torch.tensor(loss_train, device=device)
        dist.all_reduce(loss_train_tensor, op=dist.ReduceOp.SUM)
        loss_train = loss_train_tensor.item() / dist.get_world_size()

        ### EVALUATION
        vpsde.eval()
        losses_valid = []
        with torch.no_grad():
            for batch, dic in validloader:
                batch = batch.to(device)
                c = constructEmbedding(date_embedding, dic, device).to(device)

                mask_batch = mask.to(device).expand_as(batch)
                w = mask_batch.float()
                loss = vpsde.module.loss(batch, w=w, c=c)
                losses_valid.append(loss.detach())

        loss_valid = torch.stack(losses_valid).mean().item()
        loss_valid_tensor = torch.tensor(loss_valid, device=device)
        dist.all_reduce(loss_valid_tensor, op=dist.ReduceOp.SUM)
        loss_valid = loss_valid_tensor.item() / dist.get_world_size()

        # Only rank 0 logs and saves checkpoints
        if local_rank == 0:
            print(f"Epoch {epoch}: Train Loss = {loss_train}, Val Loss = {loss_valid}")
            log = {"Train Loss": loss_train, "Validation Loss": loss_valid}

            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    PATH_SAVE,
                    f"model_{epoch}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vpsde.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'loss_valid': loss_valid,
                    'date_embedding_state_dict' : date_embedding.state_dict()
                }, checkpoint_path)
                print(f"Model saved at {checkpoint_path}")

                # Log some plots
                with torch.no_grad():
                    myLoader = DataLoader(
                        validset, batch_size=10, shuffle=True, num_workers=1, persistent_workers=True
                    )
                    batch, dic = next(iter(myLoader))
                    c = constructEmbedding(date_embedding, dic, device).to(device)
                    sampled_traj = vpsde.module.sample(mask, c=c, shape=(10,), steps=128, corrections=1).detach().cpu()
                    batch = batch[0]
                    x = batch.repeat((3,) + (1,) * len(batch.shape))
                    t = torch.rand(x.shape[0], dtype=x.dtype)
                    c = constructEmbedding(date_embedding, dic, device)[0]
                    c = c.repeat((3,) + (1,) * len(c.shape))
                    # Noise levels to plot
                    t[0] = 0.5
                    t[1] = 0.9
                    t[2] = 1
                    t = t.to(device)
                    x = x.to(device)
                    c = c.to(device)
                    x_t = vpsde.module.forward(x, t, train=False)
                    print(f"x_t = {x_t.shape}, t : {t.shape}, c : {c.shape}, x : {x.shape}")
                    x_0 = vpsde.module.denoise(x_t, t, c).detach().cpu()
                    x_t = x_t.detach().cpu()
                    x = x.detach().cpu()
                path_unnorm = PATH_DATA / "train.h5"
                info = {'var_index': ['T2m', 'U10m'], 'channels': 2, 'window': 12}
                fig = plot_sample(sampled_traj, info, mask_cpu, samples=5, step=3, unnormalize=True,
                                  path_unnorm=path_unnorm)
                log['samples'] = wandb.Image(fig)

                new_tensor = torch.stack((x, x_t, x_0), dim=1).flatten(0, 1)
                fig = plot_sample(new_tensor, info, mask_cpu, samples=9, step=3, unnormalize=False,
                                  path_unnorm=path_unnorm)
                log['chart'] = wandb.Image(fig)

            wandb.log(log)

        scheduler.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

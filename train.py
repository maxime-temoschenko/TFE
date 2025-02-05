import os
import h5py
import math
import torch
#import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from pathlib import Path
from utils import TrajectoryDataset
from score import ScoreUNet, MCScoreWrapper
from score import VPSDE, MCScoreNet

PATH = Path('.')

with h5py.File(PATH / "data/mask.h5", "r") as f:
    mask = torch.tensor(f["dataset"][:], dtype=torch.float32).unsqueeze(0)  # Shape

def masked_vpsde_loss(sde, x, mask):
    w = mask.expand_as(x)
    return sde.loss(x, w=w)

CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 4096,
    'batch_size': 5,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}

trainset = TrajectoryDataset(PATH / "data/train.h5", window=10)
validset = TrajectoryDataset(PATH / "data/valid.h5", window=10)

trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=1, persistent_workers=True)
validloader = DataLoader(validset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=1, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Create the model :
window = CONFIG['window']
score = MCScoreNet(2, order=window // 2)
score.kernel = ScoreUNet(
    channels=window * 2,
    embedding=CONFIG['embedding'],
    hidden_channels=CONFIG['hidden_channels'],
    hidden_blocks=CONFIG['hidden_blocks'],
    kernel_size=CONFIG['kernel_size'],
    activation=nn.SiLU,
    spatial=2,
    padding_mode='circular',
)
'''
score_model =  MCScoreWrapper(ScoreUNet(
        channels=4,
        embedding=CONFIG["embedding"],
        hidden_channels=CONFIG["hidden_channels"],
        hidden_blocks=CONFIG["hidden_blocks"],
        activation=nn.SiLU,
    )).to(device)
'''
sde = VPSDE(score, shape=(window*2, 55, 66)).to(device)

optimizer = optim.AdamW(sde.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

# Define Learning Rate Scheduler
if CONFIG["scheduler"] == "linear":
    lr_lambda = lambda t: 1 - (t / CONFIG["epochs"])
elif CONFIG["scheduler"] == "cosine":
    lr_lambda = lambda t: (1 + math.cos(math.pi * t / CONFIG["epochs"])) / 2
elif CONFIG["scheduler"] == "exponential":
    lr_lambda = lambda t: math.exp(-7 * (t / CONFIG["epochs"]) ** 2)
else:
    raise ValueError("Invalid scheduler type")

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in (bar := trange(CONFIG["epochs"], ncols=88)):
    losses_train = []
    losses_valid = []

    ## Train
    sde.train()
    print('-------')
    print('[TRAIN LOOP]')
    for i, (batch, _) in enumerate(trainloader):
        print(f"{i}")
        batch = batch.to(device)
        optimizer.zero_grad()

        # **Apply Mask**
        mask_batch = mask.to(device).expand_as(batch)  # Expand mask to match batch size
        w = mask_batch.float()  # Convert mask to weight format

        # **Compute VPSDE Loss**
        loss = sde.loss(batch, w=w)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.detach())
    print('[\TRAIN LOOP]')
    print('-------')
    ## Validation
    sde.eval()
    with torch.no_grad():
        print('-------')
        print('[VALID LOOP]')
        for batch, _ in validloader:
            batch = batch.to(device)
            mask_batch = mask.to(device).expand_as(batch)
            w = mask_batch.float()

            loss = sde.loss(batch, w=w)
            losses_valid.append(loss)
        print('-------')
        print('[VALID LOOP]')
    ## Compute Loss Stats
    loss_train = torch.stack(losses_train).mean().item()
    loss_valid = torch.stack(losses_valid).mean().item()
    lr = optimizer.param_groups[0]['lr']

    ## Step Scheduler
    scheduler.step()

    # Save model periodically
    if (epoch + 1) % 10 == 0:
        torch.save(sde.state_dict(), PATH / f"checkpoints/model_epoch{epoch+1}.pth")
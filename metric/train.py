import sys
sys.path.append('..')
from TFE.metric.utils import *
from TFE.metric.model import *

import os
import math
import torch
import wandb
import h5py
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import tqdm
from tqdm import trange

CONFIG = {'batch_size' : 48,
          'epochs' : 1000,
          'learning_rate' : 1e-4}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using = {device}")
efficient_net = EfficientNet().to(device)

trainloader = getDataLoader('data/processed/train.h5', 'generated_samples.h5', batch_size=CONFIG['batch_size'])


wandb.init(project="efficientnet_classification", config=CONFIG)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficient_net.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in (bar := trange(CONFIG["epochs"], ncols=88)):
    efficient_net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in trainloader:
        inputs = inputs.to(device).float()  # (B, 12, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = efficient_net(inputs)  #  (B, 2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train

    scheduler.step()
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    if epoch % 10 == 0:
        torch.save(efficient_net.state_dict(), "efficientnet_finetuned.pth")







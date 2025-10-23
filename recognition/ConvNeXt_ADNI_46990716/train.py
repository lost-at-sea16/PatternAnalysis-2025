"""
Contains source code for training, validating, testing and saving the model
"""

from dataset import train_dataloader, test_dataloader
from modules import convnext_tiny, covnext_small 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import time

import wandb
import argparse
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

parser = argparse.ArgumentParser()
parser.add_argument("-c",)

criterion = parser.parse_args()
opts = criterion.c

if opts == "soft":
    crit = SoftTargetCrossEntropy()
elif opts == "label":
    crit = LabelSmoothingCrossEntropy(smoothing=0.1)
else:
    crit = nn.CrossEntropyLoss()



config = {
    "num_epochs": 60,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "architecture": "ConvNeXt-S_default",
    "criterion": opts,
    "scheduler": 5
}

num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
weight_decay = config["weight_decay"]
betas = config["betas"]
milestone = config["scheduler"]

def training(model, train_loader):
    """
    
    """
    criterion = crit
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    sched_linear = optim.lr_scheduler.LinearLR(optimizer)
    sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear, sched_cosine], milestones=[milestone])
    
    total_step = len(train_loader)

    losses = []

    model.train()
    print("> Training")
    start = time.time() 
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Loss {:5f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            epoch_loss += loss.item()
        scheduler.step()

        
        avg_loss = epoch_loss / total_step
        wandb.log({
                "train_loss": avg_loss,
                "train_lr": optimizer.param_groups[0]['lr'],
        })
        print(f"📈 Epoch {epoch+1}/{num_epochs} Complete: Avg Loss = {avg_loss:.4f}")
    
    end = time.time()
    elapsed = end-start
    print("Training took " + str(elapsed) + "secs or " + str(elapsed/60) + " mins in total")
    return model


def test(model, test_loader):
    """
    
    """
    print("testing")
    start = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test accuracy: {} %".format(100 * correct/ total))
        wandb.log({
                "test_acc": 100*correct/ total,
        })

    end = time.time()
    elapsed = end-start
    print("Testing took " + str(elapsed) + "secs or " + str(elapsed/60) + " mins")
    return model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")



train_loader = train_dataloader(batch_size)
test_loader = test_dataloader(batch_size)

wandb.init(
    entity="sophia-gleeson-the-university-of-queensland",
    project="23-10", # Set your project name
    config=config, # Log the hyperparameters
    reinit=True)


model = covnext_small().to(device)

model = training(model, train_loader)
model = test(model, test_loader)

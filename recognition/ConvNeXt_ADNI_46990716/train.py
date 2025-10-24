"""
Contains source code for training, validating, testing and saving the model
"""

from dataset import train_dataloader, test_dataloader, split_train, split_val
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

opts = parser.parse_args()
criterion = opts.c




config = {
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "weight_decay": 0.05,
    "architecture": "ConvNeXt-S_default",
    "criterion": criterion,
    "scheduler": 2,
    "smoothing":0.1,
    "drop_path_rate":0.1
}

num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
weight_decay = config["weight_decay"]
milestone = config["scheduler"]

if criterion == "soft":
    crit = SoftTargetCrossEntropy()
elif criterion == "label":
    crit = LabelSmoothingCrossEntropy(smoothing=config["smoothing"])
else:
    crit = nn.CrossEntropyLoss()

min_lr = 1e-6 


def training(model, train_loader, val_loader):
    """
    
    """
    criterion = crit
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

    #sched_linear = optim.lr_scheduler.LinearLR(optimizer)
    #sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs/2)
    sched_linear = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=milestone)
    sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - milestone, eta_min=min_lr)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear, sched_cosine], milestones=[milestone])
    
    total_step = len(train_loader)

    losses = []

    
    print("> Training")
    start = time.time() 
    for epoch in range(num_epochs):
        model.train()
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
        
        print(f"📈 Epoch {epoch+1}/{num_epochs} Complete: Avg Loss = {avg_loss:.4f}")

        print("Validating")
    
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                #forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Validation accuracy: {} %".format(100 * correct/ total))
            wandb.log({
                "train_loss": avg_loss,
                "train_lr": optimizer.param_groups[0]['lr'],
                "Val_acc": 100*correct/ total,
                "epoch": epoch,
            })


    
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



test_loader = test_dataloader(batch_size)
val_loader = split_val(batch_size)
train_loader = split_train(batch_size)

wandb.init(
    entity="sophia-gleeson-the-university-of-queensland",
    project="24-10", # Set your project name
    config=config, # Log the hyperparameters
    reinit=True)


model = covnext_small(config["drop_path_rate"]).to(device)

wandb.watch(model, log='all', log_freq=50)

model = training(model, train_loader, val_loader)
model = test(model, test_loader)

"""
Contains source code for training, validating, testing and saving the model
"""

from dataset import train_dataloader
from modules import convnext_tiny 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import time

num_epochs = 10
learning_rate = 3e-3
batch_size = 4096
weight_decay = 0.05
betas=(0.9, 0.999)

def training(model, train_loader):
    """
    
    """
    criterion = nn.CrossEntropyLoss()
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    sched_linear = optim.lr_scheduler.LinearLR(optimizer)
    sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear, sched_cosine], milestones=[20])
    
    total_step = len(train_loader)

    model.train()
    print("> Training")
    start = time.time() 
    for epoch in range(num_epochs):
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
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Loss {:5f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        scheduler.step()
    
    end = time.time()
    elapsed = end-start
    print("Training took " + str(elapsed) + "secs or " + str(elapsed/60) + " mins in total")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    

    train_loader = train_dataloader(batch_size)


    model = convnext_tiny()
    training(model, train_loader)

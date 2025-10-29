"""
Contains source code for training, validating, testing and saving the model
"""

from dataset import train_dataloader, test_dataloader, split_train, split_val, get_train_and_val
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
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
parser.add_argument("-l", type=float)
parser.add_argument("-b", type=int)
parser.add_argument("-w", type=float)
parser.add_argument("-s", type=int)
parser.add_argument("-m", type=float)
parser.add_argument("-d", type=float)

# -n 60 -l 0.0003 -b 128 -w 0.05 -s 5 -m 0.05 -d 0.0

opts = parser.parse_args()




config = {
    "num_epochs": opts.n,
    "learning_rate": opts.l,
    "batch_size": opts.b,
    "weight_decay": opts.w,
    "scheduler": opts.s,
    "smoothing": opts.m,
    "drop_path_rate":opts.d,
    "criterion": opts.c,
}

num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
weight_decay = config["weight_decay"]
milestone = config["scheduler"]


final_predictions = []
final_labels = []


def training(model, train_loader, val_loader):
    """
    
    """
    criterion = LabelSmoothingCrossEntropy(smoothing=config["smoothing"])
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

    #sched_linear = optim.lr_scheduler.LinearLR(optimizer)
    #sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs/2)
    sched_linear = optim.lr_scheduler.LinearLR(optimizer)
    sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear, sched_cosine], milestones=[milestone])
    
    total_step = len(train_loader)

    losses = []

    #ema = torch.optim.swa_utils.AveragedModel(model)

    
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
        #ema.update_parameters(model)
        
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
    return model, optimizer


def test(model, test_loader):
    """
    
    """
    

    print("Testing")
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
            
            final_predictions.extend(predicted.cpu().numpy()) 
            final_labels.extend(labels.cpu().numpy())

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


def save_model(model, optimizer, filename="convnext_best.pth"):
    """
    Saves the model's state dictionary and optimizer state.
    """
    print(f"\nSaving model to {filename}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config 
    }, filename)
    print("Model saved successfully! ✅")


# ========================================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



test_loader = test_dataloader(batch_size)
# val_loader = split_val(batch_size)
# train_loader = split_train(batch_size)

train_loader, val_loader = get_train_and_val(batch_size=batch_size)

wandb.init(
    entity="sophia-gleeson-the-university-of-queensland",
    project="24-10", # Set your project name
    config=config, # Log the hyperparameters
    reinit=True)


model = covnext_small(config["drop_path_rate"]).to(device)

wandb.watch(model, log='all', log_freq=50)

model, optimizer = training(model, train_loader, val_loader)
model = test(model, test_loader)

save_model(model, optimizer, filename="convnext_final_model_patients.pth")

# final_labels = torch.cat(label).cpu().numpy()
# final_predictions = torch.cat(predictions).cpu().numpy()

print(classification_report(final_labels, final_predictions, target_names=["NC", "AD"]))

"""
Example usage of the trained model
"""
from dataset import train_dataloader, test_dataloader, split_train, split_val
from modules import covnext_small
import matplotlib.pyplot as plt 
import torchvision
import numpy as np
import torch
import re
from sklearn.metrics import classification_report
import time

## load saved model
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = covnext_small(drop_path_rate=0.1).to(device)
    checkpoint = torch.load(r"C:\Users\sophi\OneDrive\Documents\2025\Study\sem 2\comp3710\Assignments\A3\PatternAnalysis-2025\recognition\ConvNeXt_ADNI_46990716\convnext_final_model.pth")
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    return loaded_model


test_loader = test_dataloader(128)
model = load_trained_model()

final_labels = []
final_predictions = []


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Testing")
start = time.time()
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    correct_patient = 0
    total_patient = 0
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

        
        

end = time.time()
elapsed = end-start
print("Testing took " + str(elapsed) + "secs or " + str(elapsed/60) + " mins")


print(classification_report(final_labels, final_predictions, target_names=["NC", "AD"]))
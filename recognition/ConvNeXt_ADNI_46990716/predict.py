"""
Example usage of the trained model.
Loads trained model and uses test dataset to extract 2 patients, 1 from NC and 1 from AD.

Performs classification per patient, using the 20 slices per patient.


"""
from dataset import test_dataloader, ADNIDataset, transform_test, test_set_location
from modules import covnext_small
import matplotlib.pyplot as plt 
import numpy as np
import torch
import re
import time
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

checkpoint_location = r"C:\Users\sophi\OneDrive\Documents\2025\Study\sem 2\comp3710\Assignments\A3\PatternAnalysis-2025\recognition\ConvNeXt_ADNI_46990716\convnext_final_model.pth"

## load saved model
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = covnext_small(drop_path_rate=0.1).to(device)
    checkpoint = torch.load(checkpoint_location)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    return loaded_model


test_loader = test_dataloader(128)
model = load_trained_model()

final_labels = []
final_predictions = []


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_dataset_by_patient(dataset):
    """
    Splits a dataset into a list of per-patient Subset datasets.

    Each returned Subset corresponds to one patient and contains all of their slices.

    Assumes filenames are like: '12345_01.jpeg', '12345_02.jpeg', etc.
    where the prefix before the underscore is the patient ID.

    Args:
        dataset: the dataset to split

    Returns:
        patient_datasets: list of Subset objects, one per patient
        patient_ids: list of patient IDs corresponding to each Subset
    """

    patient_id_pattern = re.compile(r"^(\d+)_.*")

    patient_indices = {}

    for i in range(len(dataset)):
        path = dataset.image_paths[i]
        filename = os.path.basename(path)
        match = patient_id_pattern.match(filename)
        if match:
            patient_id = match.group(1)
            if patient_id not in patient_indices:
                patient_indices[patient_id] = []
            patient_indices[patient_id].append(i)
        else:
            # Skip files without valid patient ID
            continue

    patient_datasets = []
    patient_ids = []

    for pid, indices in patient_indices.items():
        subset = Subset(dataset, indices)
        patient_datasets.append(subset)
        patient_ids.append(pid)

    print(f"Split into {len(patient_datasets)} patients.")
    return patient_datasets, patient_ids


test_ds = ADNIDataset(test_set_location, transform=transform_test)
patient_datasets, patient_ids = split_dataset_by_patient(test_ds)

# Inspect one patient's dataset
patient_ds = patient_datasets[0]
print(f"Patient {patient_ids[0]} has {len(patient_ds)} slices")





"""

"""

from dataset import train_dataloader, test_dataloader, split_train, split_val, ADNIDataset, train_set_location, test_set_location
from modules import covnext_small
import matplotlib.pyplot as plt 
import torchvision
import numpy as np
import torch
import re
from sklearn.metrics import classification_report
import time
import random
from torch.utils.data import Subset
import re
from collections import defaultdict


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def show_two_class_samples():
    """
    Displays one sample image from each class (NC and AD) in the test dataset.
    """
    dataloader = test_dataloader(64)
    class_names = ["NC", "AD"]

    samples = []
    sample_labels = []

    # Keep iterating until we find two different class labels
    for images, labels in dataloader:
        for i in range(len(labels)):
            img = images[i]
            lbl = labels[i].item()

            if len(samples) == 0:
                samples.append(img)
                sample_labels.append(lbl)
            elif lbl != sample_labels[0]:
                samples.append(img)
                sample_labels.append(lbl)
                break
        if len(samples) == 2:
            break

    # Plot both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f"Class: {class_names[sample_labels[i]]}")
        ax.axis("off")

    plt.show()



def count_patient_slices(folder_path):
    """
    Parses filenames in a folder to count the number of slices for each patient ID.

    The expected filename format is 'patientid_slicenumber.jpeg'.

    Args:
        folder_path (str): The path to the folder containing the files.

    Returns:
        dict: A dictionary where keys are patient IDs (str) and values are
              the count of slices (int) for that patient.
    """
    # Dictionary to store patient IDs and their slice counts
    patient_slice_counts = defaultdict(int)

    # Regex to match the desired pattern:
    # (\d+)      -> Captures one or more digits (the patient ID)
    # _          -> Matches the underscore separator
    # \d+        -> Matches one or more digits (the slice number - not captured)
    # \..*$      -> Matches the dot and any extension
    filename_pattern = re.compile(r"(\d+)_\d+\..*$")

    try:
        # Iterate over all files in the specified folder
        for filename in os.listdir(folder_path):
            # Check if the path is a file (and not a directory)
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Attempt to match the filename pattern
                match = filename_pattern.match(filename)

                if match:
                    # The patient ID is in the first (and only) capturing group
                    patient_id = match.group(1)
                    # Increment the count for this patient ID
                    patient_slice_counts[patient_id] += 1
                # Optional: Add an 'else' block here if you need to log files
                # that don't match the expected pattern
    except FileNotFoundError:
        print(f"Error: Folder not found at path: {folder_path}")
        return {} # Return an empty dictionary on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

    # Convert defaultdict back to a standard dict for the final return
    return dict(patient_slice_counts)



train_ad = count_patient_slices(train_set_location + r"\AD")
#print(results)
train_nc = count_patient_slices(train_set_location + r"\NC")
#print(results)
test_ad = count_patient_slices(test_set_location + r"\AD")
#print(results)
test_nc = count_patient_slices(test_set_location + r"\NC")
#print(results)



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

def evaluate_per_patient(model, patient_datasets, patient_ids, device, batch_size=128):
    """
    Evaluate model performance per patient by averaging slice predictions,
    and compute ROC + AUC for AD vs NC classification.

    Args:
        model: trained PyTorch model
        patient_datasets (list[Subset]): per-patient datasets
        patient_ids (list[str]): corresponding patient IDs
        device: torch device
        batch_size (int): dataloader batch size
        log_to_wandb (bool): whether to log metrics and plots to Weights & Biases

    Returns:
        results (dict): {patient_id: {"true": int, "pred": int, "prob": float}}
        metrics (dict): accuracy, precision, recall, f1, auc
    """
    model.eval()
    results = {}

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for pid, ds in zip(patient_ids, patient_datasets):
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            probs = []
            labels = []

            for images, lbls in loader:
                images = images.to(device)
                outputs = model(images)
                probs.extend(outputs.softmax(1)[:, 1].cpu().numpy())  # Probability for AD class
                labels.extend(lbls.cpu().numpy())

            mean_prob = np.mean(probs)
            pred_label = 1 if mean_prob >= 0.5 else 0
            true_label = int(np.round(np.mean(labels)))  # all slices same label ideally

            results[pid] = {"true": true_label, "pred": pred_label, "prob": mean_prob}

            all_true.append(true_label)
            all_pred.append(pred_label)
            all_prob.append(mean_prob)

    # --- Compute metrics ---
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(all_true, all_pred, target_names=["NC", "AD"], output_dict=True)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(all_true, all_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Per-Patient ROC Curve (AD vs NC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Aggregate metrics
    metrics = {
        "accuracy": report["accuracy"],
        "precision_NC": report["NC"]["precision"],
        "recall_NC": report["NC"]["recall"],
        "precision_AD": report["AD"]["precision"],
        "recall_AD": report["AD"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "auc": roc_auc
    }

    # Print report + confusion matrix
    print("\n=== Per-Patient Classification Report ===")
    print(classification_report(all_true, all_pred, target_names=["NC", "AD"]))
    print("Confusion Matrix:\n", cm)
    print(f"AUC: {roc_auc:.3f}")



    return results, metrics

def plot_patient_slices(model, patient_dataset, patient_label_name, device):
    """
    Plots 20 slices of a single patient with model predictions and actual labels.
    """
    model.eval()
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(len(patient_dataset)):
            image, label = patient_dataset[i]
            image = image.unsqueeze(0).to(device)  # add batch dim
            output = model(image)
            pred = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1)[0, pred].item()

            img = image.cpu().squeeze().numpy()

            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Pred: {'AD' if pred==1 else 'NC'} ({prob:.2f})\nTrue: {patient_label_name}")

    fig.suptitle(f"Predictions and Actual Labels for {patient_label_name} Patient", fontsize=16)
    plt.tight_layout()
    plt.show()



test_ds = ADNIDataset(test_set_location, transform=transform_test)
patient_datasets, patient_ids = split_dataset_by_patient(test_ds)

# Inspect one patient's dataset
patient_ds = patient_datasets[0]
print(f"Patient {patient_ids[0]} has {len(patient_ds)} slices")


results, metrics = evaluate_per_patient(model, patient_datasets, patient_ids, device)

print(f"\n✅ Patient-Level Accuracy: {metrics['accuracy']:.3f}")
print(f"✅ Patient-Level AUC: {metrics['auc']:.3f}")

# Find one AD and one NC patient
ad_patient = next(ds for ds in patient_datasets if ds[0][1] == 1)
nc_patient = next(ds for ds in patient_datasets if ds[0][1] == 0)

plot_patient_slices(model, ad_patient, "AD", device)
plot_patient_slices(model, nc_patient, "NC", device)



"""
Contains dataloader for loading and preprocessing data
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import re

train_set_location = r"/home/groups/comp3710/ADNI/AD_NC/train"
test_set_location = r"/home/groups/comp3710/ADNI/AD_NC/test"

class ADNIDataset(Dataset):
    """
    ADNI PyTorch Dataset for loading 2D MRI image slices.

    This dataset handles images categorized as Alzheimer's disease (AD) and normal control (NC), 
    indexing images based on their folder structure.
    It returns images as grayscale (L mode) PIL objects or transformed Tensors.
    """


    def __init__(self, root_dir, transform=None):
        """
        Initializes the ADNI Dataset by indexing all image files.

        Args:
            root_dir (string): Directory containing the images. Path should be in the structure
                    ~/ADNI/AD_NC/train or ~/ADNI/AD_NC/test
            transform (callable, optional): Optional transform to be applied on a sample

        https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/custom-datasets-pytorch
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ["AD", "NC"]
        self.class_to_idx = {"NC":0, "AD":1}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

# transforms used for the train dataset
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# transforms used for the test dataset
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def train_dataloader(batch_size):
    """
    Creates a PyTorch DataLoader for the train set of the ADNI dataset

    Args:
        batch_size (int): Number of samples per batch

    Returns:
        train dataloader (Dataloader): Dataloader for the train set
    """
    dataset = ADNIDataset(root_dir=train_set_location, transform=transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_dataloader(batch_size):
    """
    Creates a PyTorch DataLoader for the test set of the ADNI dataset

    Args:
        batch_size (int): Number of samples per batch

    Returns:
        test dataloader (Dataloader): Dataloader for the test set
    """
    dataset = ADNIDataset(root_dir=test_set_location, transform=transform_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



def split_train(batch_size, val_split=0.2, seed=60):  
    dataset = ADNIDataset(root_dir=train_set_location, transform=transform_train)
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, _ = random_split(dataset, [train_size, val_size], generator=generator)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def split_val(batch_size, val_split=0.2, seed=60):  
    dataset = ADNIDataset(root_dir=train_set_location, transform=transform_test)
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def split_dataset_by_patient(dataset, validation_split=0.2, random_seed=60):
    """
    Splits the test dataset into a test and validation set at patient level

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing images
        validation_split (float, optional): Fraction of unique patients to include in the validation set. Defaults to 0.2.
        random_seed (int, optional): Seed for reproducible shuffling of patient IDs. Defaults to 60.

    Returns:
        tuple: A tuple containing:
            - test_subset (torch.utils.data.Subset): Subset of the dataset for testing.
            - val_subset (torch.utils.data.Subset): Subset of the dataset for validation.
    """

    patient_id_pattern = re.compile(r"^(\d+)_.*")

    patient_indicies = {}

    for i in range(len(dataset)):
        path = dataset.image_paths[i]
        filename = os.path.basename(path)

        match = patient_id_pattern.match(filename)
        if match:
            patient_id = match.group(1)
            if patient_id not in patient_indicies:
                patient_indicies[patient_id] = []
            
            patient_indicies[patient_id].append(i)
        else:
            # file doesn't match the expected format -> skip
            continue
    
    patient_ids = list(patient_indicies.keys())
    random.seed(random_seed)
    random.shuffle(patient_ids) # Shuffle the list of unique patient IDs

    num_val_patients = int(len(patient_ids) * validation_split)
    
    val_patient_ids = patient_ids[:num_val_patients]
    test_patient_ids = patient_ids[num_val_patients:]
    
    test_indices = []
    val_indices = []

    for id in test_patient_ids:
        test_indices.extend(patient_indicies[id])

    for id in val_patient_ids:
        val_indices.extend(patient_indicies[id])

    test_subset = Subset(dataset, test_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"Total unique patients: {len(patient_ids)}")
    print(f"Train patients: {len(test_patient_ids)} ({len(test_subset)} slices)")
    print(f"Validation patients: {len(val_patient_ids)} ({len(val_subset)} slices)")
    
    return test_subset, val_subset

def get_test_and_val(batch_size, val_split=0.2, seed=60):
    """
    Prepares DataLoaders for the test and validation subsets of the ADNI dataset, 
    splitting at the patient level to prevent leakage.

    Args:
        batch_size (int): Number of samples per batch for the DataLoaders.
        val_split (float, optional): Fraction of patients to include in the validation set. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility in splitting. Defaults to 60.

    Returns:
        tuple: A tuple containing:
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    """
    dataset = ADNIDataset(root_dir=test_set_location, transform=transform_test)

    test_subset, val_subset = split_dataset_by_patient(dataset, val_split)

    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return test_loader, val_loader
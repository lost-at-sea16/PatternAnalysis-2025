"""
Contains dataloader for loading and preprocessing data
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms as transforms
import os
from PIL import Image
import timm.data.transforms_factory as transforms_factory

# train_set_location = r"H:\comp3710\ADNI\AD_NC\train"
# test_set_location = r"H:\comp3710\ADNI\AD_NC\test"

# train_set_location = r"C:\Users\sophi\OneDrive\Documents\2025\Study\sem 2\comp3710\Assignments\A3\ADNI\AD_NC\train"
# test_set_location = r"C:\Users\sophi\OneDrive\Documents\2025\Study\sem 2\comp3710\Assignments\A3\ADNI\AD_NC\test"

train_set_location = r"/home/groups/comp3710/ADNI/AD_NC/train"
test_set_location = r"/home/groups/comp3710/ADNI/AD_NC/test"

class ADNIDataset(Dataset):
    """ADNI dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the images
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
    


# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
    
#     transforms.Normalize(mean=[0.5], std=[0.5]),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

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

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def train_dataloader(batch_size):
    dataset = ADNIDataset(root_dir=train_set_location, transform=transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_dataloader(batch_size):
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

"""
Contains dataloader for loading and preprocessing data
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image

train_set_location = r"H:\comp3710\ADNI\AD_NC\train"
test_set_location = r"H:\comp3710\ADNI\AD_NC\test"

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

        self.samples = []
        self.classes = ["AD", "NC"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



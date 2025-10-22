import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

class DegradationDataset(Dataset):
    def __init__(self, root_dir, dataset_type='rain', split='train', transform=None):
        """
        Dataset for rain/haze/low-light.
        - root_dir: Path to dataset folder (e.g., 'data/rain100h', 'data/reside', 'data/lol')
        - dataset_type: 'rain' (Rain100L/H), 'haze' (RESIDE), 'lowlight' (LOL)
        - split: 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.dataset_type = dataset_type
        self.transform = transform
        self.image_pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        if self.dataset_type in ['rain', 'haze', 'lowlight']:
            degraded_dir = os.path.join(self.root_dir, 'rain')  # Low-light/hazy/rainy images stored in 'rain'
            gt_dir = os.path.join(self.root_dir, 'norain')
            for file in sorted(os.listdir(degraded_dir)):
                degraded_path = os.path.join(degraded_dir, file)
                gt_path = os.path.join(gt_dir, file)  # Assume same filename
                pairs.append((degraded_path, gt_path))
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        degraded_path, gt_path = self.image_pairs[idx]
        degraded = Image.open(degraded_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        if self.transform:
            degraded = self.transform(degraded)
            gt = self.transform(gt)
        return degraded, gt

# Augmentation and normalization
transform = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loaders
def get_rain_loader(root_dir, split='train', batch_size=8):
    dataset = DegradationDataset(root_dir, 'rain', split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

def get_haze_loader(root_dir, split='train', batch_size=8):
    dataset = DegradationDataset(root_dir, 'haze', split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

def get_lowlight_loader(root_dir, split='train', batch_size=8):
    dataset = DegradationDataset(root_dir, 'lowlight', split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

def get_combined_loader(rain_root, haze_root, split='train', batch_size=8):
    rain_dataset = DegradationDataset(rain_root, 'rain', split, transform)
    haze_dataset = DegradationDataset(haze_root, 'haze', split, transform)
    combined = torch.utils.data.ConcatDataset([rain_dataset, haze_dataset])
    return DataLoader(combined, batch_size=batch_size, shuffle=True)
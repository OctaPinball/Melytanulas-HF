import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score
import h5py
import cv2
import numpy as np
import os
import tifffile as tiff

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, subset_size=None, transform=None):
        self.h5_file = h5_file
        self.data = h5py.File(h5_file, "r")
        self.transform = transform

        total_size = self.data["image"].shape[0]
        if subset_size:
            self.indices = np.random.choice(total_size, subset_size, replace=False)
        else:
            self.indices = np.arange(total_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.data["image"][actual_idx]
        label = self.data["label"][actual_idx]

        # Remove extra dimension if present
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
        if label.ndim == 3 and label.shape[-1] == 2:  # One-hot encoding detected
            label = np.argmax(label, axis=-1)

        # Add channel dimension to the image
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)  # [C, H, W]
        label = torch.tensor(label, dtype=torch.long)     # [H, W]

        return {"image": image, "label": label}
    
def compute_mean_and_std(dataset):
    """
    Compute the mean and standard deviation of the dataset.

    Args:
        dataset: Dataset object (e.g., HDF5Dataset)

    Returns:
        tuple: mean (mu), standard deviation (sd)
    """
    sum_pixels = 0
    sum_squared_pixels = 0
    num_pixels = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]["image"]  # Assuming dataset returns image and label
        pixels = sample.numpy().flatten()  # Flatten to a 1D array

        sum_pixels += pixels.sum()
        sum_squared_pixels += (pixels ** 2).sum()
        num_pixels += pixels.size

    mu = sum_pixels / num_pixels
    variance = (sum_squared_pixels / num_pixels) - (mu ** 2)
    sd = np.sqrt(variance)

    return mu, sd

def debug_dataset(dataset):
    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(f"Index {idx}: Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")
        if idx >= 5:  # Limit the debug output to a few samples
            break
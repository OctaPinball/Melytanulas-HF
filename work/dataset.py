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
    def __init__(self, h5_file, subset_size=None):
        self.h5_file = h5_file
        self.data = h5py.File(h5_file, 'r')
        
        # Get the total size of the dataset
        total_size = self.data['image'].shape[0]
        
        # If subset_size is specified, use only that many samples
        if subset_size:
            self.indices = np.random.choice(total_size, subset_size, replace=False)
            self.indices.sort()  # Ensure indices are in increasing order
            self.mean, self.std = self.calculate_mean_std()  # Calculate mean and std for the subset
        else:
            self.indices = np.arange(total_size)
            self.mean, self.std = self.get_mean_std()  # Use pre-calculated mean and std
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.data['image'][actual_idx]
        label = self.data['label'][actual_idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_mean_std(self):
        if 'train.mean' in self.data and 'train.sd' in self.data:
            return self.data["train.mean"][()], self.data["train.sd"][()]
        else:
            raise ValueError("Mean and standard deviation are not stored in the dataset.")

    def calculate_mean_std(self):
        images = self.data['image'][self.indices]  # Only use the subset of the data
        mean = np.mean(images)
        std = np.std(images)
        return mean, std
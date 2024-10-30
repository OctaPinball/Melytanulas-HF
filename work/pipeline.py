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

import unet
import dataset as ds
import evaluate as eval
import parameters as params
import train
import dataloader as dl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Preprocess




### Training

## Unet training (Baseline)
dataset = ds.HDF5Dataset(params.DATA_PATH, subset_size=5)
dataloader = dl.CombinedDataLoader(dataset, params.VAL_SPLIT_RATIO, 4)
unet = unet.CNNModel().to(device)
train.train(unet, dataloader, 1000, device, 5, 5, "unet")

## n1 Model training

## n2 Model training

## n3 Model training

## n4 Model training


### Evaluate
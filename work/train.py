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

import dataloader as dl
from parameters import FILE_PATH, LOG_PATH, MODEL_PATH
import evaluate as eval


def train(model, model_name: str, dataloader: dl.CombinedDataLoader, max_epoch: int, device, save_interval: int = None, evaluate_interval: int = None):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training loop with validation
    for epoch in range(1, max_epoch):
        model.train()
        running_loss = 0.0
        
        # Training
        for images, labels in dataloader.train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.permute(0, 3, 1, 2))
            
            outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
            labels = torch.argmax(labels, dim=-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch}/{max_epoch}], Train Loss: {running_loss / len(dataloader.train_loader)}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in dataloader.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.permute(0, 3, 1, 2))
                
                outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
                labels = torch.argmax(labels, dim=-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Epoch [{epoch}/{max_epoch}], Validation Loss: {val_loss / len(dataloader.val_loader)}")

        # Save model
        if save_interval is not None and epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}_epoch_{epoch}.pth"))

        # Evaluate model
        if evaluate_interval is not None and epoch % evaluate_interval == 0:
            mu, sd = dataloader.dataset.get_mean_std()
            eval.predict(FILE_PATH, model_name, model, mu, sd, device)
            eval.Score(FILE_PATH, model_name, LOG_PATH)



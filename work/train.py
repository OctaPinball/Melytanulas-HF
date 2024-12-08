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
from tqdm import tqdm  # Progress bar library
import time
from monai.losses import DiceLoss


def train_model(model, model_name, train_loader, val_loader, device, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DiceLoss(sigmoid=False, softmax=True)

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Training
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(device)  # Shape: [B, C, H, W]
            labels = batch["label"].to(device)  # Shape: [B, H, W]

            labels = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()

            optimizer.zero_grad()
            outputs = model(images)  # Expected shape: [B, num_classes, H, W]
            if isinstance(outputs, list):
                outputs = outputs[0]  # Take the primary tensor output

            # Compute the loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)  # Shape: [B, C, H, W]
                labels = batch["label"].to(device)  # Shape: [B, H, W]

                labels = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()

                outputs = model(images)  # Model output: [B, num_classes, H, W]
                if isinstance(outputs, list):
                    outputs = outputs[0]  # Take the primary tensor output

                # Compute the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}_{epoch}.pth"))
            print("Saved the best model.")

    return model

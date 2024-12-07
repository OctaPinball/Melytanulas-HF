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
from evaluate import Score, predict
from tqdm import tqdm  # Progress bar library
import time


def train(model, model_name: str, dataloader: dl.CombinedDataLoader, max_epoch: int, device, save_interval: int = None, evaluate_interval: int = None):
    print(f"Starting training in model: {model_name}")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()  # Start the timer

    print(f"Training started on device: {device}")
    print(f"Total epochs: {max_epoch}\n")

    # Training loop with validation
    for epoch in range(1, max_epoch):
        print(f"Epoch [{epoch}/{max_epoch}]")
        model.train()
        running_loss = 0.0
        train_loader_len = len(dataloader.train_loader)
        
        # Training
        print("Training:")
        train_bar = tqdm(dataloader.train_loader, total=train_loader_len, desc="Train Progress", ncols=80)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.permute(0, 3, 1, 2))
            
            outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
            labels = torch.argmax(labels, dim=-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.set_postfix(Loss=f"{loss.item():.4f}")
        
        avg_train_loss = running_loss / train_loader_len
        print(f"Train Loss: {avg_train_loss:.4f}")

        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loader_len = len(dataloader.val_loader)
        print("\nValidation:")
        val_bar = tqdm(dataloader.val_loader, total=val_loader_len, desc="Validation Progress", ncols=80)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.permute(0, 3, 1, 2))
                
                outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
                labels = torch.argmax(labels, dim=-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_bar.set_postfix(Loss=f"{loss.item():.4f}")
        
        avg_val_loss = val_loss / val_loader_len
        print(f"Validation Loss: {avg_val_loss:.4f}\n")
        
        print(f"Epoch [{epoch}/{max_epoch}], Validation Loss: {val_loss / len(dataloader.val_loader)}")

        # Save model
        if save_interval is not None and epoch % save_interval == 0:
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            save_path = os.path.join(LOG_PATH, f"{model_name}_epoch_{epoch}.pth" if model_name else f"model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Evaluate model
        if evaluate_interval is not None and epoch % evaluate_interval == 0:
            print("Running evaluation...")
            mu, sd = dataloader.dataset.get_mean_std()
            predict(dir_path=FILE_PATH, model_name=model_name, CNN_model=model, mu=mu, sd=sd, device=device)
            Score(dir_path=FILE_PATH, model_name=model_name, log_path=LOG_PATH)
            print("Evaluation completed.\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")


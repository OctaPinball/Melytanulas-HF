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

def predict(dir_path, CNN_model, model_name: str, mu=0, sd=1, device='cuda'):
    # Input size
    n1 = 112

    # Get all the files for testing
    files = os.listdir(dir_path)
    files.remove("log")
    files.remove("Training.h5")

    CNN_model.eval()

    for file in files:
        print(f"Segmenting: {os.path.join(file)}")

        # Get image shape and number of slices
        temp = cv2.imread(os.path.join(dir_path, file, "data", "slice001.tiff"), cv2.IMREAD_GRAYSCALE)
        number_of_slices = len(os.listdir(os.path.join(dir_path, file, "data")))

        # Determine cropping coordinates
        midpoint = temp.shape[0] // 2
        n11, n12 = midpoint - int(n1 / 2), midpoint + int(n1 / 2)

        # Initialize input array
        input1 = np.zeros(shape=[number_of_slices, n1, n1])

        # Loading data
        for n in range(number_of_slices):
            input_filename = f"slice{n + 1:03}.tiff"
            ImageIn = cv2.imread(os.path.join(dir_path, file, "data", input_filename), cv2.IMREAD_GRAYSCALE)
            ImageIn = (ImageIn - mu) / sd
            input1[n, :, :] = ImageIn[n11:n12, n11:n12]

        # Making predictions
        output = np.zeros(shape=[number_of_slices, 2, n1, n1])  # Correct shape with 2 channels
        for n in range(number_of_slices):
            img_tensor = torch.tensor(input1[n, :, :], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = CNN_model(img_tensor)

                # Apply softmax to get probabilities
                out = F.softmax(out, dim=1)

                # Resize the output to match the expected size (112, 112)
                out = F.interpolate(out, size=(n1, n1), mode='bilinear', align_corners=False)

                # Assign the result directly
                output[n, :, :, :] = out.cpu().numpy()

        # Perform argmax over the class dimension (axis 1)
        output = np.argmax(output, 1)
        # Writing data to output
        for n in range(number_of_slices):
            Imout = np.zeros(shape=[temp.shape[0], temp.shape[1]])
            Imout[n11:n12, n11:n12] = output[n, :, :]
            output_filename = f"slice{n + 1:03}.tiff"
            directory_to_check = os.path.join(dir_path, file, "auto segmentation", model_name)
            if not os.path.exists(directory_to_check):
                os.makedirs(directory_to_check)
            cv2.imwrite(os.path.join(dir_path, file, "auto segmentation", model_name, output_filename), np.uint8(255 * Imout))

def Score(dir_path, model_name: str, log_path):
    # Create a txt file to write results
    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f1_scores = []

        files = os.listdir(dir_path)
        files.remove("log")
        files.remove("Training.h5")

        for file in files:
            pred_dir = os.path.join(dir_path, file, "auto segmentation", model_name)
            true_dir = os.path.join(dir_path, file, "cavity")

            # Get predicted and true masks
            pred = []
            true = []
            for k in range(len(os.listdir(pred_dir))):
                pred_img_path = os.path.join(pred_dir, f"slice{k+1:03}.tiff")
                true_img_path = os.path.join(true_dir, f"slice{k+1:03}.tiff")

                # Use tifffile to open the TIFF files
                pred_img = tiff.imread(pred_img_path) // 255
                true_img = tiff.imread(true_img_path) // 255

                pred.append(pred_img.flatten())
                true.append(true_img.flatten())

            pred = np.concatenate(pred)
            true = np.concatenate(true)

            # Calculate F1 score
            f1 = f1_score(true, pred, average="binary")
            f.write(f"{file} - F1 Score: {round(f1, 3)}\n")
            f1_scores.append(f1)

        f.write(f"\nOVERALL F1 AVERAGE = {np.mean(f1_scores)}\n\n")

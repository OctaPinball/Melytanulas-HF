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
    files.remove("model")
    files.remove("Training.h5")

    files = sorted(files)

    files = files[:1]

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

                out = CNN_model(img_tensor)  # Model output: [B, num_classes, H, W]
                if isinstance(out, list):
                    out = out[0]  # Take the primary tensor output

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

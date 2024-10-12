import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import h5py
import cv2
import numpy as np
import os
import tifffile as tiff


### Helper Functions
def predict(dir_path, CNN_model, mu=0, sd=1, device='cuda'):
    # Input size
    n1 = 112

    # Get all the files for testing
    files = os.listdir(dir_path)
    files.remove("log")
    files.remove("Utah_Training.h5")

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
            cv2.imwrite(os.path.join(dir_path, file, "auto segmentation", output_filename), 255 * Imout)




def Score(dir_path, log_path):
    # Create a txt file to write results
    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f1_scores = []

        files = os.listdir(dir_path)
        files.remove("log")
        files.remove("Utah_Training.h5")

        for file in files:
            pred_dir = os.path.join(dir_path, file, "auto segmentation")
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


### Dataset Class
class HDF5Dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.data = h5py.File(h5_file, 'r')

    def __len__(self):
        return self.data['image'].shape[0]

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        label = self.data['label'][idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_mean_std(self):
        return self.data["train.mean"][()], self.data["train.sd"][()]


### Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, output_padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        # Decoder
        dec4_out = self.dec4(enc5_out)
        dec3_out = self.dec3(dec4_out)
        dec2_out = self.dec2(dec3_out)
        dec1_out = self.dec1(dec2_out)

        return self.final(dec1_out)


### Training
file_path = "UTAH Test set"
log_path = "UTAH Test set/log"
data_path = "UTAH Test set/Utah_Training.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = HDF5Dataset(data_path)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, 1001):
    model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images.permute(0, 3, 1, 2))  # Add channel dimension

        # Convert one-hot labels to class indices
        outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
        labels = torch.argmax(labels, dim=-1)  # Convert from [batch_size, 112, 112, 2] to [batch_size, 112, 112]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch}/1000], Loss: {running_loss / len(data_loader)}")

    # Save checkpoint and evaluate
    torch.save(model.state_dict(), os.path.join(log_path, f"model_epoch_{epoch}.pth"))
    mu, sd = dataset.get_mean_std()
    predict(file_path, model, mu, sd, device)
    Score(file_path, log_path)

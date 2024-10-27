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

# Define validation split ratio
VAL_SPLIT_RATIO = 0.2  # 20% for validation

### Helper Functions
def predict(dir_path, CNN_model, mu=0, sd=1, device='cuda'):
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
            cv2.imwrite(os.path.join(dir_path, file, "auto segmentation", output_filename), np.uint8(255 * Imout))





def Score(dir_path, log_path):
    # Create a txt file to write results
    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f1_scores = []

        files = os.listdir(dir_path)
        files.remove("log")
        files.remove("Training.h5")

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
        """Retrieve pre-calculated mean and std from the HDF5 dataset."""
        if 'train.mean' in self.data and 'train.sd' in self.data:
            return self.data["train.mean"][()], self.data["train.sd"][()]
        else:
            raise ValueError("Mean and standard deviation are not stored in the dataset.")

    def calculate_mean_std(self):
        """Calculate mean and std for the subset of the dataset."""
        images = self.data['image'][self.indices]  # Only use the subset of the data
        mean = np.mean(images)
        std = np.std(images)
        return mean, std

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
file_path = "../data/preprocessed_data"
log_path = "../data/preprocessed_data/log"
data_path = '../data/preprocessed_data/Training.h5'
dataset = HDF5Dataset(data_path, subset_size=5)

# Split dataset into training and validation subsets
train_size = int((1 - VAL_SPLIT_RATIO) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create separate dataloaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current working directory: {os.getcwd()}")
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop with validation
for epoch in range(1, 1001):
    model.train()
    running_loss = 0.0
    
    # Training
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images.permute(0, 3, 1, 2))
        
        outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
        labels = torch.argmax(labels, dim=-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch}/1000], Train Loss: {running_loss / len(train_loader)}")
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.permute(0, 3, 1, 2))
            
            outputs = F.interpolate(outputs, size=(112, 112), mode='bilinear', align_corners=False)
            labels = torch.argmax(labels, dim=-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch [{epoch}/1000], Validation Loss: {val_loss / len(val_loader)}")

    # Save model and run evaluation every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(log_path, f"model_epoch_{epoch}.pth"))
        mu, sd = dataset.get_mean_std()
        predict(file_path, model, mu, sd, device)
        Score(file_path, log_path)
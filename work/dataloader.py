from torch.utils.data import random_split, DataLoader
import dataset as ds
from dataset import HDF5Dataset

def get_hdf5_data_loaders(h5_file_path, batch_size, val_split=0.2, subset_size=None):
    full_dataset = HDF5Dataset(h5_file_path, subset_size=subset_size)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
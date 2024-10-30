from torch.utils.data import random_split, DataLoader
import dataset as ds

class CombinedDataLoader():
    def __init__(self, dataset: ds.HDF5Dataset, val_split_ratio: float, batch_size: int):
        self.dataset = dataset
        train_size = int((1 - val_split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
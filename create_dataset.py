import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.transforms import Compose

class VSDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data
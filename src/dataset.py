import os
import numpy as np
from torch.utils.data import Dataset
import torch

class TrafficImageDataset(Dataset):
    def __init__(self, tensor_dir, label):
        self.tensor_dir = tensor_dir
        self.files = [f for f in os.listdir(tensor_dir) if f.endswith('.npy')]
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.tensor_dir, self.files[idx])
        img = np.load(path)
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
        label = torch.tensor(self.label, dtype=torch.float32)
        return img, label

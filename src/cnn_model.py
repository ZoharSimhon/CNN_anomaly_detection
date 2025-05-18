import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MAX_PACKETS_PER_FLOW

class TrafficCNN(nn.Module):
    def __init__(self):
        super(TrafficCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Compute the size after two poolings
        reduced_size = MAX_PACKETS_PER_FLOW // 4  # each pooling divides size by 2
        self.flat_features = 64 * reduced_size * reduced_size

        self.fc1 = nn.Linear(self.flat_features, 128)
        self.fc2 = nn.Linear(128, 1) # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 32, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 64, H/4, W/4)
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))        # Output in [0, 1]
        return x

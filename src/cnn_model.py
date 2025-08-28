import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMAGE_SIZE

class SimpleCNN(nn.Module):
    """
    Simple CNN for flow images.
    Output: logits (no softmax) so that energy-based OOD scoring can be applied.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        reduced_size = IMAGE_SIZE // 4
        self.fc1 = nn.Linear(64 * reduced_size * reduced_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 32, H/2, W/2)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 64, H/4, W/4)
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                   # raw scores (logits)
        return logits

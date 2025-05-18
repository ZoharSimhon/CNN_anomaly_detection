# test.py

import torch
from torch.utils.data import DataLoader
from cnn_model import TrafficCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    model = TrafficCNN().to(device)
    model.load_state_dict(torch.load(MODEL_DIR))
    model.eval()

    # Load benign and malicious sets
    benign_dataset = TrafficImageDataset(f"{TENSORS_DIR}/test/benign", BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(f"{TENSORS_DIR}/test/malicious", MALICIOUS_LABEL)
    loader = DataLoader(benign_dataset + malicious_dataset, batch_size=1, shuffle=False)

    TP = TN = FP = FN = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            out = model(x)
            pred = (out > 0.5).float()

            if pred == 1 and y == 1: TP += 1
            elif pred == 0 and y == 0: TN += 1
            elif pred == 1 and y == 0: FP += 1
            elif pred == 0 and y == 1: FN += 1

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

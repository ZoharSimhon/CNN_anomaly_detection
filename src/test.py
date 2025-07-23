import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_DIR, map_location=device))
    model.eval()

    benign_dataset = TrafficImageDataset(TEST_BENIGN_DIR, BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(TEST_MALICIOUS_DIR, MALICIOUS_LABEL)
    full_dataset = benign_dataset + malicious_dataset
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

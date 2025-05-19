import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from cnn_model import TrafficCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    model = TrafficCNN().to(device)
    model.load_state_dict(torch.load(MODEL_DIR))
    model.eval()

    # Load benign and malicious datasets
    benign_dataset = TrafficImageDataset(TEST_BENIGN_DIR, BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(TEST_MALICIOUS_DIR, MALICIOUS_LABEL)

    # Combine datasets
    full_dataset = benign_dataset + malicious_dataset
    loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            prob = output.item()
            print(f"Model prediction probability: {prob:.4f}")
            prediction = int(prob > 0.3)  # Threshold for classification

            y_true.append(y.item())
            y_pred.append(prediction)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Compute detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))

    # Optional: Print TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score

import numpy as np

from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_energy_score(logits):
    """
    Compute energy score for OOD detection.
    Reference: https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
    """
    return -T * torch.logsumexp(logits / T, dim=1)


def test_model():
    # load model
    model = SimpleCNN(num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_DIR, map_location=device))
    model.eval()

    benign_dataset = TrafficImageDataset(TEST_BENIGN_DIR, BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(TEST_MALICIOUS_DIR, MALICIOUS_LABEL)
    full_dataset = benign_dataset + malicious_dataset
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    scores, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            energy = compute_energy_score(logits)
            scores.extend(energy.cpu().numpy())
            labels.extend(y.numpy())  # use the true labels here

    scores = np.array(scores)
    labels = np.array(labels)

    # Normalize scores (optional but helps stability)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

   # Threshold for classification
    preds = (scores > OOD_THRESHOLD).astype(int)

    # Metrics
    # auroc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)

    print("=== Evaluation Results ===")
    # print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Benign", "Malicious"]))

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
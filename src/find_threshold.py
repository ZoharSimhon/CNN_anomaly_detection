import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_curve, precision_recall_curve

from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_energy_score(logits):
    return -logits.squeeze()

def find_optimal_threshold():
    print(f"🔎 Analyzing Model to find Optimal Threshold...")
    
    # 1. Load Model & Data
    model = SimpleCNN(num_classes=1).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR, map_location=device))
    model.eval()

    benign_dataset = TrafficImageDataset(config.TEST_BENIGN_DIR, config.BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(config.TEST_MALICIOUS_DIR, config.MALICIOUS_LABEL)
    
    full_dataset = ConcatDataset([benign_dataset, malicious_dataset])
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    energy_scores = []
    true_labels = []

    # 2. Get Scores
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            energy = compute_energy_score(logits)
            energy_scores.extend(energy.cpu().numpy())
            true_labels.extend(y.numpy())

    energy_scores = np.array(energy_scores)
    true_labels = np.array(true_labels)

    # 3. Analyze Distributions
    benign_energies = energy_scores[true_labels == 0]
    mal_energies = energy_scores[true_labels == 1]

    print("\n=== Energy Statistics ===")
    print(f"Benign    | Mean: {benign_energies.mean():.4f} | Max: {benign_energies.max():.4f} | Min: {benign_energies.min():.4f}")
    print(f"Malicious | Mean: {mal_energies.mean():.4f} | Max: {mal_energies.max():.4f} | Min: {mal_energies.min():.4f}")

    # 4. Find Threshold for 95% TPR (True Positive Rate)
    # We want to catch 95% of attacks. What threshold does that?
    fpr, tpr, thresholds = roc_curve(true_labels, energy_scores)
    
    # Find index where TPR is closest to 0.95
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    optimal_threshold = thresholds[idx]
    
    print(f"\n=== Recommendation ===")
    print(f"To detect {target_tpr*100}% of attacks, set OOD_THRESHOLD to: {optimal_threshold:.4f}")
    print(f"At this threshold, your False Positive Rate (FPR) will be: {fpr[idx]:.4f}")
    
    # 5. Check separation
    if benign_energies.mean() > mal_energies.mean():
        print("\n❌ CRITICAL WARNING: Benign Energy is HIGHER than Malicious Energy.")
        print("The model has learned the WRONG direction. Check your labels or loss function.")

if __name__ == "__main__":
    find_optimal_threshold()
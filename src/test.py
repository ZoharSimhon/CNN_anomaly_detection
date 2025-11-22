import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os

from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
import config
from results import evaluate_model  # Import the new module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_energy_score(logits):
    """
    Compute energy score for Binary Energy-Based Model.
    Energy E(x) = -logits (since we trained Benign to be +logits)
    """
    return -logits.squeeze()

def test_model():
    print(f"Testing model on device: {device}")
    
    # 1. Load Model
    # Use num_classes=1 for scalar output
    model = SimpleCNN(num_classes=1).to(device)
    
    if not os.path.exists(config.MODEL_DIR):
        print(f"Error: Model file not found at {config.MODEL_DIR}")
        return

    try:
        model.load_state_dict(torch.load(config.MODEL_DIR, map_location=device))
    except RuntimeError:
        print("Error: Model architecture mismatch. Ensure num_classes=1 used in training.")
        return
        
    model.eval()

    # 2. Load Data
    try:
        benign_dataset = TrafficImageDataset(config.TEST_BENIGN_DIR, config.BENIGN_LABEL)
        malicious_dataset = TrafficImageDataset(config.TEST_MALICIOUS_DIR, config.MALICIOUS_LABEL)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"Loaded: {len(benign_dataset)} Benign, {len(malicious_dataset)} Malicious samples.")
    
    full_dataset = ConcatDataset([benign_dataset, malicious_dataset])
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    energy_scores = []
    true_labels = []

    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            logits = model(x)
            energy = compute_energy_score(logits)
            
            energy_scores.extend(energy.cpu().numpy())
            true_labels.extend(y.numpy())

    energy_scores = np.array(energy_scores)
    true_labels = np.array(true_labels)

    # 4. Calculate and Print Results
    evaluate_model(true_labels, energy_scores, config.OOD_THRESHOLD)

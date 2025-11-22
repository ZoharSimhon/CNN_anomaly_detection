# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os

from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def energy_loss_fn(logits, labels):
    """
    Implements Equation (6) from Liu et al. NeurIPS 2020 for binary case.
    L = E_in[max(0, E(x) - m_in)^2] + E_out[max(0, m_out - E(x))^2]
    
    Mapping:
    - logits (f(x)): High values -> Benign
    - Energy E(x) = -logits
    """
    # 1. Compute Energy: E(x) = -logits
    energy = -logits.squeeze()

    # 2. Split into Benign (ID) and Attack (OE)
    benign_mask = (labels == BENIGN_LABEL)
    oe_mask = (labels == MALICIOUS_LABEL)
    
    energy_benign = energy[benign_mask]
    energy_oe = energy[oe_mask]
    
    total_loss = 0.0
    
    # 3. Minimize Energy for Benign (Penalty if E > m_in)
    if len(energy_benign) > 0:
        # "Push Benign Energy DOWN"
        loss_in = torch.pow(torch.relu(energy_benign - M_IN), 2).mean()
        total_loss += loss_in

    # 4. Maximize Energy for Outliers (Penalty if E < m_out)
    if len(energy_oe) > 0:
        # "Push Attack Energy UP"
        loss_out = torch.pow(torch.relu(M_OUT - energy_oe), 2).mean()
        total_loss += OE_LAMBDA * loss_out  # Weight the outlier term

    return total_loss

def train_model():
    print(f"Training with Energy-Bounded Learning (Liu et al. 2020) on {device}...")
    
    # Ensure directories exist
    if not os.path.exists(TRAIN_BENIGN_DIR) or not os.path.exists(TRAIN_OE_DIR):
        print(f"ERROR: Please ensure '{TRAIN_BENIGN_DIR}' and '{TRAIN_OE_DIR}' exist and contain .npy files.")
        return

    # 1. Load Datasets
    # Benign Data (In-Distribution)
    dataset_benign = TrafficImageDataset(TRAIN_BENIGN_DIR, BENIGN_LABEL)
    # Attack Data (Outlier Exposure - The "One Class" you mentioned)
    dataset_oe = TrafficImageDataset(TRAIN_OE_DIR, MALICIOUS_LABEL)
    
    print(f"Benign Samples: {len(dataset_benign)}")
    print(f"Outlier Exposure Samples: {len(dataset_oe)}")
    
    # Combine them. DataLoader will shuffle them together.
    full_dataset = ConcatDataset([dataset_benign, dataset_oe])
    loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    # We use num_classes=1 because we are learning a scalar "Normality Score"
    model = SimpleCNN(num_classes=1).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        benign_seen = 0
        oe_seen = 0
        
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            logits = model(x)
            
            # Compute Custom Energy Loss
            loss = energy_loss_fn(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Statistics
            benign_seen += (y == BENIGN_LABEL).sum().item()
            oe_seen += (y == MALICIOUS_LABEL).sum().item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f} | Mix: {benign_seen} Benign, {oe_seen} OE")

    # Save Model
    torch.save(model.state_dict(), MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")
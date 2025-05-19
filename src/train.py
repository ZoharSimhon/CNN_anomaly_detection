import torch
from torch.utils.data import DataLoader

from cnn_model import TrafficCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    model = TrafficCNN().to(device)
    dataset = TrafficImageDataset(f"{TENSORS_DIR}/train/", BENIGN_LABEL)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), MODEL_DIR)

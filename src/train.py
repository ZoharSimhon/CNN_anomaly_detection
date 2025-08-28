import torch
from torch.utils.data import DataLoader
from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    model = SimpleCNN(num_classes=1).to(device)  # only benign
    benign_dataset = TrafficImageDataset(TRAIN_DIR, BENIGN_LABEL)
    loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()  # binary one-class

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, _ in loader:  # labels are always 0 for benign
            x = x.to(device)
            y = torch.zeros(x.size(0), 1).to(device)  # all benign

            out = model(x)  # shape: (B,1)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_DIR)


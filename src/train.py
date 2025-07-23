import torch
from torch.utils.data import DataLoader
from cnn_model import SimpleCNN
from dataset import TrafficImageDataset
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    model = SimpleCNN(num_classes=2).to(device)

    benign_dataset = TrafficImageDataset(TRAIN_BENIGN_DIR, BENIGN_LABEL)
    malicious_dataset = TrafficImageDataset(TRAIN_MALICIOUS_DIR, MALICIOUS_LABEL)
    print(f"Benign dataset size: {len(benign_dataset)}")
    print(f"Malicious dataset size: {len(malicious_dataset)}")
    dataset = benign_dataset + malicious_dataset
    # dataset = benign_dataset  # For simplicity, only using benign data for now
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)  # (B, 2)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_DIR)

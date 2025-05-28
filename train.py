import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import MelSpectrogramDataset
from cfg import CLASSES
from model import CNNClassifier


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct = 0.0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


# Training loop
def main():
    # Config
    data_dir = "data/processed"
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    val_split = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    full_dataset = MelSpectrogramDataset(data_dir=data_dir, classes=CLASSES)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = CNNClassifier(num_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
        )

    # Save
    torch.save(model.state_dict(), "cnn_classifier.pth")


if __name__ == "__main__":
    main()

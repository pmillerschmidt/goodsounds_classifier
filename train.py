import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MelSpectrogramDataset
from model import CNNClassifier
from cfg import CLASSES


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        # update loss
        total_loss += loss.item() * X.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def main():
    data_dir = "data/processed"
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    val_split = 0.2
    patience = 5
    weight_decay = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # complete dataset
    full_dataset = MelSpectrogramDataset(
        data_dir=data_dir, classes=CLASSES, training=False
    )
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    # train/val datasets
    train_samples = [full_dataset.samples[i] for i in train_subset.indices]
    val_samples = [full_dataset.samples[i] for i in val_subset.indices]
    train_ds = MelSpectrogramDataset(
        data_dir, CLASSES, training=True, sample_list=train_samples
    )
    val_ds = MelSpectrogramDataset(
        data_dir, CLASSES, training=False, sample_list=val_samples
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    # Model
    model = CNNClassifier(num_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    # Training loop with early stopping
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # return stats
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
        )
        # update learning rate
        scheduler.step(val_loss)
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "deep_cnn_classifier.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. Best Val Acc: {best_val_acc:.2%}")
                break


if __name__ == "__main__":
    main()

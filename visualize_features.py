import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import umap.umap_ as umap
from torch.utils.data import DataLoader
from dataset import MelSpectrogramDataset
from model import CNNClassifier
from cfg import CLASSES, MODEL_PATH

# Config
DATA_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 5000
REDUCTION_METHOD = "pca"

# Load model
model = CNNClassifier(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Hook for intermediate features
features = []
labels = []


def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())


handle = model.classifier[0].register_forward_hook(hook_fn)

# Load data
dataset = MelSpectrogramDataset(data_dir=DATA_DIR, classes=CLASSES, training=False)
sample_indices = np.random.choice(
    len(dataset), size=min(MAX_SAMPLES, len(dataset)), replace=False
)
sampler = torch.utils.data.Subset(dataset, sample_indices)
dataloader = DataLoader(sampler, batch_size=32, shuffle=False)

# Forward pass
all_labels = []
with torch.no_grad():
    for X, y in dataloader:
        X = X.to(DEVICE)
        _ = model(X)
        all_labels.extend(y.numpy())

# Prepare feature matrix
features = np.concatenate(features, axis=0)
labels = np.array(all_labels)

if REDUCTION_METHOD == "umap":
    # Dimensionality reduction
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    features_2d = reducer.fit_transform(features)
elif REDUCTION_METHOD == "pca":
    # Dimensionality reduction with PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
else:
    raise ValueError(f"Invalid reduction method: {REDUCTION_METHOD}")

# Fit SVM
clf = SVC(kernel="linear")
clf.fit(features_2d, labels)

# Meshgrid for decision boundary
h = 0.5
x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.2)
for i, cls in enumerate(CLASSES):
    idxs = labels == i
    x_coords = features_2d[idxs, 0]
    y_coords = features_2d[idxs, 1]
    plt.scatter(x_coords, y_coords, label=cls, s=25, edgecolors="k", alpha=0.7)

    # Filter out outliers and label near center of cluster
    if len(x_coords) > 5:
        dists = np.linalg.norm(
            np.stack([x_coords, y_coords], axis=1)
            - np.mean(np.stack([x_coords, y_coords], axis=1), axis=0),
            axis=1,
        )
        keep = dists < np.percentile(dists, 90)
        x_center = np.mean(x_coords[keep])
        y_center = np.mean(y_coords[keep])
        plt.text(
            x_center,
            y_center,
            cls,
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.7
            ),
        )

if REDUCTION_METHOD == "umap":
    plt.title("UMAP Embedding of CNN Features with SVM Decision Boundary")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
elif REDUCTION_METHOD == "pca":
    plt.title("PCA Embedding of CNN Features with SVM Decision Boundary")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
else:
    raise ValueError(f"Invalid reduction method: {REDUCTION_METHOD}")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

handle.remove()

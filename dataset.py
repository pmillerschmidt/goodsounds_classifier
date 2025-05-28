import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None):
        """
        Args:
            data_dir (str): Path to processed data directory
            classes (list): List of instrument class names (str)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    path = os.path.join(cls_dir, fname)
                    samples.append((path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)
        mel = np.expand_dims(mel, axis=0)  # Add channel dimension: [1, H, W]
        mel = torch.tensor(mel, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            mel = self.transform(mel)

        return mel, label

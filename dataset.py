import os
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentation import apply_spec_augment


class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, classes, training=True, sample_list=None):
        self.data_dir = data_dir
        self.classes = classes
        self.training = training
        if sample_list is not None:
            self.samples = sample_list
        else:
            self.samples = []
            for label, cls in enumerate(classes):
                cls_dir = os.path.join(data_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.endswith(".npy"):
                        self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)
        if self.training:
            mel = apply_spec_augment(mel)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return mel_tensor, label

import numpy as np
import random


def time_mask(spec, max_width=10, num_masks=1):
    """Apply time masking to a mel spectrogram."""
    spec = spec.copy()
    num_frames = spec.shape[1]

    for _ in range(num_masks):
        start = random.randint(0, max(0, num_frames - max_width))
        width = random.randint(1, max_width)
        spec[:, start : start + width] = 0
    return spec


def freq_mask(spec, max_bins=8, num_masks=1):
    """Apply frequency masking to a mel spectrogram."""
    spec = spec.copy()
    num_mels = spec.shape[0]

    for _ in range(num_masks):
        start = random.randint(0, max(0, num_mels - max_bins))
        width = random.randint(1, max_bins)
        spec[start : start + width, :] = 0
    return spec


def apply_spec_augment(spec, time_max_width=10, freq_max_bins=8):
    """Apply both time and frequency masking."""
    spec = time_mask(spec, max_width=time_max_width, num_masks=1)
    spec = freq_mask(spec, max_bins=freq_max_bins, num_masks=1)
    return spec

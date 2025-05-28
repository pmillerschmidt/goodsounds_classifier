import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict
from cfg import SOURCE_DIR, OUTPUT_DIR, SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, N_FFT


def preprocess_audio(file_path):
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:
        y = y[:, 0]  # mono

    y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(DURATION * SAMPLE_RATE)
    if len(y) < target_len:
        y = librosa.util.fix_length(data=y, size=target_len)
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 40) / 40  # normalize to [0, 1]
    return mel_db.astype(np.float32)


def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for instrument in os.listdir(SOURCE_DIR):
        input_dir = os.path.join(SOURCE_DIR, instrument)
        if not os.path.isdir(input_dir):
            continue

        output_dir = os.path.join(OUTPUT_DIR, instrument)
        os.makedirs(output_dir, exist_ok=True)

        for fname in tqdm(os.listdir(input_dir), desc=f"Processing {instrument}"):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(input_dir, fname)
            try:
                mel = preprocess_audio(fpath)
                name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, name + ".npy"), mel)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")


if __name__ == "__main__":
    process_dataset()

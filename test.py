import os
import torch
import librosa
import numpy as np
import soundfile as sf
from train import CNNClassifier
from cfg import CLASSES

# Config
MODEL_PATH = "cnn_classifier.pth"
SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024
INPUT_WAV = "example.wav"  # Replace with your test file path


# Preprocess single file into mel spectrogram
def preprocess_audio(path):
    y, sr = sf.read(path)
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
    mel_tensor = (
        torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # shape: [1, 1, H, W]
    return mel_tensor


def predict(path, model, device):
    model.eval()
    x = preprocess_audio(path).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
        top_idx = torch.argmax(probs).item()
        print(f"Prediction: {CLASSES[top_idx]} ({probs[top_idx]:.2%})")

        print("\nTop-5 class probabilities:")
        topk = torch.topk(probs, k=min(5, len(CLASSES)))
        for idx, prob in zip(topk.indices, topk.values):
            print(f"{CLASSES[idx]}: {prob:.2%}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    if not os.path.exists(INPUT_WAV):
        raise FileNotFoundError(f"{INPUT_WAV} not found.")

    predict(INPUT_WAV, model, device)


if __name__ == "__main__":
    main()

import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import gradio as gr
from train import CNNClassifier
from cfg import (
    CLASSES,
    MODEL_PATH,
    SAMPLE_RATE,
    DURATION,
    N_MELS,
    HOP_LENGTH,
    N_FFT,
)


# Preprocessing
def preprocess_audio(y, sr):
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
    mel_norm = (mel_db + 40) / 40  # normalize
    return (
        mel_norm.astype(np.float32),
        mel_db,
    )  # return both normalized and raw for display


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# Inference function
def classify_instrument(audio):
    y, sr = sf.read(audio)
    mel, mel_db = preprocess_audio(y, sr)
    input_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidences = {CLASSES[i]: float(p) for i, p in enumerate(probs)}

    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title(f"Mel Spectrogram - Predicted: {pred_class}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    return pred_class, confidences, fig


# Gradio app
app = gr.Interface(
    fn=classify_instrument,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.Text(label="Predicted Instrument"),
        gr.Label(label="Confidence Scores"),
        gr.Plot(label="Mel Spectrogram"),
    ],
    title="Instrument Classifier",
    description="Upload a short (1.5s) audio clip of a single instrument and get the predicted instrument class.",
)

if __name__ == "__main__":
    app.launch()

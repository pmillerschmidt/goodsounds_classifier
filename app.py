import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import gradio as gr

from model import CNNClassifier
from cfg import CLASSES, SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, N_FFT, MODEL_PATH


# Preprocess function
def preprocess_audio(y, sr):
    if len(y.shape) > 1:
        y = y[:, 0]
    y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(DURATION * SAMPLE_RATE)
    y = (
        librosa.util.fix_length(data=y, size=target_len)
        if len(y) < target_len
        else y[:target_len]
    )

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 40) / 40
    mel_tensor = torch.tensor(mel_norm[np.newaxis, :, :], dtype=torch.float32)
    return mel_tensor, mel_db


# Saliency map
def compute_saliency(model, input_tensor, target_class):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    return saliency


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Inference function
def classify_instrument(audio_path):
    y, sr = sf.read(audio_path)
    mel_tensor, mel_db = preprocess_audio(y, sr)
    mel_tensor = mel_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidences = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
    # Saliency map
    saliency = compute_saliency(model, mel_tensor.clone(), pred_idx)
    # Plot mel spectrogram and saliency map
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax1,
    )
    ax1.set_title("Mel Spectrogram")
    ax2.imshow(saliency, aspect="auto", origin="lower")
    ax2.set_title("Saliency Map")
    ax2.set_ylabel("Mel Bins")
    ax2.set_xlabel("Time Frames")

    plt.tight_layout()
    return confidences, fig


# Gradio UI layout with inputs on the left and plot on the right
with gr.Blocks() as app:
    gr.Markdown(
        """## Instrument Classifier
    Upload a short audio clip to predict the instrument class.
    """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            scores = gr.Label(label="Confidence Scores")
        with gr.Column():
            plot_output = gr.Plot(label="Spectrogram & Saliency")

    audio_input.change(
        fn=classify_instrument,
        inputs=audio_input,
        outputs=[scores, plot_output],
    )

if __name__ == "__main__":
    app.launch()

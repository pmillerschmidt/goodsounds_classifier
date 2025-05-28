MODEL_PATH = "cnn_classifier.pth"
SOURCE_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024

CLASSES = [
    "bass",
    "trumpet",
    "oboe",
    "sax",
    "trumpet",
    "cello",
    "flute",
    "violin",
    "piccolo",
    "clarinet",
]

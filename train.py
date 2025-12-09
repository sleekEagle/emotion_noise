from datasets import load_dataset, Audio
import torchaudio
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import pandas as pd


# Map emotion labels to IDs
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for i, label in enumerate(emotion_labels)}

# Load pretrained Wav2Vec2
model_name = "facebook/wav2vec2-base"  # or "facebook/wav2vec2-large"

# Initialize feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# Initialize model with classification head
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(emotion_labels),
    label2id=label2id,
    id2label=id2label,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    final_dropout=0.1,
)

print(f"Model loaded: {model_name}")
print(f"Number of labels: {len(emotion_labels)}")

data_path = "C:\\Users\\lahir\\code\\CREMA-D\\AudioWAV"
splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/MahiA/CREMA-D/" + splits["train"])

# Load emotion dataset (CREMA-D as example)
dataset = load_dataset("jhartwell/crema-d")

# Check dataset structure
print(dataset)
print(dataset['train'][0])



def prepare_dataset(batch):
    # Load audio
    audio = batch["audio"]
    
    # Convert to array if needed
    if isinstance(audio, dict):
        waveform = audio["array"]
        sampling_rate = audio["sampling_rate"]
    else:
        waveform = audio
    
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        waveform = resampler(torch.tensor(waveform)).numpy()
    
    batch["audio"] = {"array": waveform, "sampling_rate": 16000}
    
    # Convert emotion label to ID
    if "emotion" in batch:
        batch["labels"] = label2id.get(batch["emotion"].lower(), -1)
    
    return batch

# Apply preprocessing
dataset = dataset.map(prepare_dataset, remove_columns=["emotion"])


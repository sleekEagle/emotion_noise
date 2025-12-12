from datasets import Dataset, Audio, ClassLabel
import os
from transformers import AutoFeatureExtractor
import torch
import numpy as np

data_dir = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
MAX_LEN = 16000
SR = 16000

EMOTION_MAP = {
    "ANG": 0,  # angry
    "DIS": 1,  # disgust
    "FEA": 2,  # fearful
    "HAP": 3,  # happy
    "NEU": 4,  # neutral
    "SAD": 5,  # sad
}

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=MAX_LEN, truncation=True
    )
    return inputs

def get_hf_dataset():
    paths, labels, sub = get_paths()
    subject_names = ClassLabel(names=sorted(set(sub)))
    ds = Dataset.from_dict({"audio": paths, "label": labels, "sub": sub})
    hf_dataset = ds.cast_column("audio", Audio())
    hf_dataset = hf_dataset.cast_column("label", ClassLabel(names=sorted(set(labels))))
    hf_dataset = hf_dataset.cast_column("sub", ClassLabel(names=sorted(set(sub))))
    hf_dataset = hf_dataset.map(preprocess_function, remove_columns="audio", batched=True)
    hf_dataset = hf_dataset.map(lambda x: {"label": int(x["label"])})  # Ensure it's Python int
    hf_dataset.set_format(type="torch", columns=["input_values", "label", "sub"])  # Set torch format
    # hf_dataset = hf_dataset.map(lambda example: {'label': int(example['label'])})
    # labels_list = sorted(set(hf_dataset["label"]))
    # hf_dataset = hf_dataset.cast_column("label", ClassLabel(names=labels_list))

    # sorted_names = [name for name, _ in sorted(EMOTION_MAP.items(), key=lambda x: x[1])]
    # class_label = ClassLabel(names=sorted_names)
    # hf_dataset = hf_dataset.cast_column("label", class_label)

    #seperate splits
    train_subjects, eval_subjects = create_splits()

    train_ds = hf_dataset.filter(lambda row: row["sub"] in train_subjects)
    eval_ds = hf_dataset.filter(lambda row: row["sub"] in eval_subjects)

    return train_ds, eval_ds, subject_names.names


# get_hf_dataset()

def get_paths():
    paths = []
    labels = []
    sub = []

    for file in os.listdir(data_dir):
        label = file.split('_')[2]
        sub.append(file.split('_')[0])
        paths.append(os.path.abspath(os.path.join(data_dir, file)))
        labels.append(label)
    return paths, labels, sub

def create_splits():
    paths, labels, sub = get_paths()
    subjects = list(set(sub))
    subjects.sort()
    train_subjects = subjects[: int(0.8 * len(subjects))]
    eval_subjects  = subjects[int(0.8 * len(subjects)) :]
    return train_subjects, eval_subjects

def get_split_paths():
    train_subjects, eval_subjects = create_splits()
    train_paths = []
    eval_paths = []

    for file in os.listdir(data_dir):
        subject = file.split('_')[0]
        full_path = os.path.abspath(os.path.join(data_dir, file))
        if subject in train_subjects:
            train_paths.append(full_path)
        else:
            eval_paths.append(full_path)
    return train_paths, eval_paths





if __name__ == "__main__":
    get_split_paths()

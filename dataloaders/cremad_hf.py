from datasets import Dataset, Audio, ClassLabel
import os
from transformers import AutoFeatureExtractor

data_dir = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

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
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

def get_hf_dataset():
    paths = []
    labels = []
    sub = []

    for file in os.listdir(data_dir):
        label = file.split('_')[2]
        sub.append(file.split('_')[0])
        paths.append(os.path.abspath(os.path.join(data_dir, file)))
        labels.append(label)

    ds = Dataset.from_dict({"audio": paths, "label": labels, "sub": sub})
    hf_dataset = ds.cast_column("audio", Audio())
    hf_dataset = hf_dataset.cast_column("label", ClassLabel(names=sorted(set(labels))))
    hf_dataset = hf_dataset.cast_column("sub", ClassLabel(names=sorted(set(sub))))
    hf_dataset = hf_dataset.map(preprocess_function, remove_columns="audio", batched=True)

    sorted_names = [name for name, _ in sorted(EMOTION_MAP.items(), key=lambda x: x[1])]
    class_label = ClassLabel(names=sorted_names)
    hf_dataset = hf_dataset.cast_column("label", class_label)

    #seperate splits
    ids = sorted(list(set(hf_dataset['sub'])))
    ids.sort()
    train_subjects = ids[: int(0.8 * len(ids))]
    eval_subjects  = ids[int(0.8 * len(ids)) :]

    train_ds = hf_dataset.filter(lambda row: row["sub"] in train_subjects)
    eval_ds = hf_dataset.filter(lambda row: row["sub"] in eval_subjects)

    return train_ds, eval_ds


# get_hf_dataset()

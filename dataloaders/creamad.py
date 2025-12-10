import torch
from torch.utils.data import Dataset as T_Dataset
from torch.utils.data import DataLoader
from datasets import Dataset
from torchvision import transforms
import os
import librosa
from transformers import Wav2Vec2Processor
from transformers import DataCollatorWithPadding
from transformers import AutoFeatureExtractor


EMOTION_MAP = {
    "ANG": 0,  # angry
    "DIS": 1,  # disgust
    "FEA": 2,  # fearful
    "HAP": 3,  # happy
    "NEU": 4,  # neutral
    "SAD": 5,  # sad
}

class CremaDataset(T_Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        samples = []
        
        # Load your data files here
        for filename in os.listdir(data_dir):
            if filename.endswith(('.wav')):
                samples.append(filename)

        actors = list(set([s.split('_')[0] for s in samples]))
        actors.sort()
        num_actors = len(actors)
        if split == 'train':
            selected_actors = actors[:int(0.8 * num_actors)]
        elif split == 'val':
            selected_actors = actors[int(0.8 * num_actors):]

        self.samples = [s for s in samples if s.split('_')[0] in selected_actors]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load audio/data
        file_name = self.samples[idx]
        sample_path = os.path.join(self.data_dir, file_name)

        signal, sr = librosa.load(sample_path, sr=None)
        signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
        # inputs = self.feature_extractor(
        #     signal, sampling_rate=self.feature_extractor.sampling_rate, max_length=16000, truncation=True
        # )

        emotion_code = file_name.split('_')[2]
        label = EMOTION_MAP[emotion_code]
        # processed = self.processor(signal, sampling_rate=16000, return_tensors="pt")

        item = {
            "audio": signal,
            "sampling_rate": self.feature_extractor.sampling_rate,
            "labels": label,
        }

        return item


def get_dataset(data_dir, batch_size=32, shuffle=True, num_workers=0, split='train'):
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    dataset = CremaDataset(data_dir, split=split)
    hf_dataset = Dataset.from_list([dataset[i] for i in range(len(dataset))])

    # collator = DataCollatorWithPadding(tokenizer=processor)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     num_workers=num_workers
    # )
    return hf_dataset

# dl = get_dataset(r'C:\Users\lahir\code\CREMA-D\AudioWAV', batch_size=3, split='train')
# pass

# for i, batch in enumerate(dl):
#     print(i, batch)
#     break
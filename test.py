from dataloaders.cremad_hf import get_eval_details, get_all_ds, EMOTION_MAP, feature_extractor, MAX_LEN, SR
import torch
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor
import torchaudio
from mix_noise import mix_noise_np
from datasets import load_dataset, Audio
from speaker_sep import AudioSeperator
from match_speaker import SpeakerMatcher
import soundfile as sf
import os



accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

label2id, id2label = dict(), dict()
for i, label in enumerate(EMOTION_MAP.keys()):
    label2id[label] = str(i)
    id2label[str(i)] = label

#dataloader
# eval_ds, subject_names = get_eval_details(r'C:\Users\lahir\code\CREMA-D\AudioWAV')
eval_ds, subject_names = get_all_ds(r'C:\Users\lahir\code\CREMA-D\clean')

dataloader = torch.utils.data.DataLoader(
    eval_ds, 
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: {
        'input_values': torch.stack([torch.tensor(item['input_values']) for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch]),
        'sub': torch.stack([item['sub'] for item in batch])
    }
)

num_labels = len(EMOTION_MAP)
model = AutoModelForAudioClassification.from_pretrained(
    r'C:\\Users\\lahir\\models\\emo_1s\\checkpoint-4600\\', num_labels=num_labels, label2id=label2id, id2label=id2label
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''

clean data: acc = 0.49
speech noisy data: acc = 0.448
cleaned data: 0.2601   we might need to train on cleaned data for better results
'''
def eval_model():
    with torch.no_grad():
        pred_list,gt_list=[],[]
        for i,batch in enumerate(dataloader):
            print(f'Processing batch {i+1}/{len(dataloader)}')
            inputs = batch['input_values']
            labels = batch['labels']
            outputs = model(input_values=inputs).logits
            pred = torch.argmax(outputs, dim=-1)
            pred_list.append(pred.item())
            gt_list.append(labels.item())   
    acc = accuracy.compute(predictions=pred_list, references=gt_list)
    print(f'Accuracy: {acc["accuracy"]:.4f}')


'''
acc = 0.4509
'''
def eval_model_noise():
    with torch.no_grad():
        pred_list,gt_list=[],[]
        for i,batch in enumerate(dataloader):
            print(f'Processing batch {i+1}/{len(dataloader)}')
            inputs = batch['input_values']
            inputs = inputs.squeeze(0).cpu().numpy()
            noisy = mix_noise_np(inputs, 16000, r'C:\Users\lahir\data\noise\speech_noise.mp3', snr_db=15)
            inputs = torch.tensor(noisy).unsqueeze(0)


            labels = batch['labels']
            outputs = model(input_values=inputs).logits
            pred = torch.argmax(outputs, dim=-1)
            pred_list.append(pred.item())
            gt_list.append(labels.item())   
    acc = accuracy.compute(predictions=pred_list, references=gt_list)
    print(f'Accuracy: {acc["accuracy"]:.4f}')

noise_path = r'C:\Users\lahir\data\noise\speech_noise.mp3'
tmp_noise = r'C:\Users\lahir\data\noise\tmp.wav'
sep_path = r'C:\Users\lahir\data\noise\sep\output.wav'
sep_dir = os.path.dirname(sep_path)

def eval_model_denoise():
    ap = AudioSeperator()
    # sm = SpeakerMatcher(ref_audio_path = )
    ref_audio_paths = get_ref_audio()
    
    with torch.no_grad():
        pred_list,gt_list=[],[]
        for i,batch in enumerate(dataloader):
            print(f'Processing batch {i+1}/{len(dataloader)}')
            inputs = batch['input_values']
            inputs = inputs.squeeze(0).cpu().numpy()
            noisy = mix_noise_np(inputs, SR, noise_path, snr_db=15)
            sf.write(tmp_noise, noisy, SR)

            #remove noise
            ap.process_file(
                input_path = tmp_noise,
                output_path = sep_path
            )

            sub = batch['sub'][0].item()
            sub_name = subject_names[sub]
            ref_path = ref_audio_paths[sub_name]
            matcher = SpeakerMatcher(ref_audio_path=ref_path, similarity_threshold=0.6)
            
            candidate_paths = [os.path.join(sep_dir, f) for f in os.listdir(sep_dir) if f.endswith('.wav')]
            best_match, best_sim = matcher.find_best_match(candidate_paths)

            labels = batch['labels']
            outputs = model(input_values=inputs).logits
            pred = torch.argmax(outputs, dim=-1)
            pred_list.append(pred.item())
            gt_list.append(labels.item())   
    acc = accuracy.compute(predictions=pred_list, references=gt_list)
    print(f'Accuracy: {acc["accuracy"]:.4f}')





if __name__ == "__main__":
    eval_model()









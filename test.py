from dataloaders.cremad_hf import get_hf_dataset, EMOTION_MAP, feature_extractor
import torch
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor

from datasets import load_dataset, Audio


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

label2id, id2label = dict(), dict()
for i, label in enumerate(EMOTION_MAP.keys()):
    label2id[label] = str(i)
    id2label[str(i)] = label

#dataloader
train_ds, eval_ds = get_hf_dataset()
dataloader = torch.utils.data.DataLoader(
    eval_ds, 
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: {
        'input_values': torch.stack([torch.tensor(item['input_values']) for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch])
    }
)

num_labels = len(EMOTION_MAP)
model = AutoModelForAudioClassification.from_pretrained(
    r'C:\\Users\\lahir\\models\\emo\\checkpoint-4600\\', num_labels=num_labels, label2id=label2id, id2label=id2label
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def eval_model_noise():
    with torch.no_grad():
        pred_list,gt_list=[],[]
        for i,batch in enumerate(dataloader):
            print(f'Processing batch {i+1}/{len(dataloader)}')
            inputs = batch['input_values']
            inputs = inputs + torch.randn_like(inputs) * 0.1
            labels = batch['labels']
            outputs = model(input_values=inputs).logits
            pred = torch.argmax(outputs, dim=-1)
            pred_list.append(pred.item())
            gt_list.append(labels.item())   
    acc = accuracy.compute(predictions=pred_list, references=gt_list)
    print(f'Accuracy: {acc["accuracy"]:.4f}')


if __name__ == "__main__":
    eval_model_noise()









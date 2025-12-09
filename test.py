'''

take splits from 
https://huggingface.co/datasets/MahiA/CREMA-D

model
https://huggingface.co/Supreeta03/wav2vec2-base-CREMAD-sentiment-analysis

'''
# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os


path = r'C:\Users\lahir\data\cremad\CREMA-D\audios\1001_IEO_DIS_MD.wav'
pipe = pipeline("audio-classification", model="Supreeta03/wav2vec2-base-CREMAD-sentiment-analysis")

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/MahiA/CREMA-D/" + splits["test"])


gt_list, pred_list = [], []
for i, row in df.iterrows():
    print(f'Processing {i} / {len(df)}', end='\r')
    file = row['path'].split('/')[1]
    path =  os.path.join(data_path, file)
    result = pipe(path)
    pred = result[0]['label'][:3].lower()
    gt = file.split('_')[2].lower()
    pred_list.append(pred)
    gt_list.append(gt)

print("Accuracy:", accuracy_score(gt_list, pred_list))
print(classification_report(gt_list, pred_list))
print(confusion_matrix(gt_list, pred_list))




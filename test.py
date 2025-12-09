# Use a pipeline as a high-level helper
from transformers import pipeline
path = r'C:\Users\lahir\data\cremad\CREMA-D\audios\1001_IEO_DIS_MD.wav'
pipe = pipeline("audio-classification", model="Supreeta03/wav2vec2-base-CREMAD-sentiment-analysis")
pipe(path)


import pandas as pd
splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/MahiA/CREMA-D/" + splits["test"])

import os
data_path = "C:\\Users\\lahir\\code\\CREMA-D\\AudioWAV"
for i, row in df.iterrows():
    file = row['path'].split('/')[1]
    path =  os.path.join(data_path, file)
    result = pipe(path)
    print(f"Actual label: {file.split('_')[2]} Predicted label: {result[0]['label']}, Score: {result[0]['score']:.4f}")



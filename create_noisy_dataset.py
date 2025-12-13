import os
import mix_noise
import soundfile as sf
from dataloaders.cremad_hf import get_split_paths

data_path = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
noise_path = r'C:\Users\lahir\data\noise\speech_noise.mp3'
output_path = r'C:\Users\lahir\code\CREMA-D\white_speech_noise_db15_train'
os.makedirs(output_path, exist_ok=True)


train_paths, eval_paths = get_split_paths(data_path)
for i,file in enumerate(train_paths):
    out_file = os.path.join(output_path, os.path.basename(file))
    if os.path.exists(out_file):
        continue
    print(f'Processing file {i+1}/{len(train_paths)}')
    noisy = mix_noise.mix_noise(file, noise_path, snr_db=15, use_white_noise=True)
    sf.write(out_file, noisy, 16000)


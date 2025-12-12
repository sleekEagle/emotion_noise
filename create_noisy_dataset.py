import os
import mix_noise
import soundfile as sf
from dataloaders.cremad_hf import get_split_paths

data_path = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
noise_path = r'C:\Users\lahir\data\noise\speech_noise.mp3'
output_path = r'C:\Users\lahir\code\CREMA-D\speech_noise_db15'
os.makedirs(output_path, exist_ok=True)


train_paths, eval_paths = get_split_paths()
for i,file in enumerate(eval_paths):
    print(f'Processing file {i+1}/{len(eval_paths)}')
    noisy = mix_noise.mix_noise(file, noise_path, snr_db=15)
    out_file = os.path.join(output_path, os.path.basename(file))
    sf.write(out_file, noisy, 16000)


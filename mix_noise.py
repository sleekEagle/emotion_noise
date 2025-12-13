import os
import librosa
import numpy as np  
import soundfile as sf


snr_db = 15

def mix_noise(audio_path, noise_path, snr_db=15, use_white_noise=False):
    speech, sr_speech = librosa.load(audio_path, sr=None)
    noise, sr_noise = librosa.load(noise_path, sr=sr_speech)

    #clip the noise to match the length of the speech
    s = speech.size
    max_start = noise.size - s - 1
    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + s
    noise_segment = noise[start_idx:end_idx]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_segment * scaling_factor
    noisy_speech = speech + scaled_noise

    if use_white_noise:
        white_noise = np.random.normal(0, np.sqrt(target_noise_power), len(speech))
        noisy_speech += white_noise
    
    return noisy_speech


def mix_noise_np(speech, sr_speech, noise_path, snr_db=15):
    noise, sr_noise = librosa.load(noise_path, sr=sr_speech)

    #clip the noise to match the length of the speech
    s = speech.size
    max_start = noise.size - s - 1
    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + s
    noise_segment = noise[start_idx:end_idx]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_segment * scaling_factor
    noisy_speech = speech + scaled_noise
    
    return noisy_speech

# noise_path = r'C:\Users\lahir\data\noise\test.mp3'
# audio_path = r'C:\Users\lahir\code\CREMA-D\AudioWAV\1001_DFA_SAD_XX.wav'
# librosa.output.write_wav('noisy_speech.wav', noisy_speech, sr_speech)
# sf.write('noisy_speech.wav', noisy_speech, sr_speech)





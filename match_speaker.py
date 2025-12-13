from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
import speaker_sep
import shutil

class SpeakerMatcher:
    def __init__(self, ref_audio_path, similarity_threshold=0.75):
        self.encoder = VoiceEncoder()
        self.ref_embed = self.encoder.embed_utterance(preprocess_wav(Path(ref_audio_path)))
        self.threshold = similarity_threshold
    
    def find_matches(self, candidates):
        matches = []
        for audio_file in candidates:
            wav = preprocess_wav(Path(audio_file))
            embed = self.encoder.embed_utterance(wav)
            similarity = np.dot(self.ref_embed, embed) / (
                np.linalg.norm(self.ref_embed) * np.linalg.norm(embed)
            )
            if similarity > self.threshold:
                matches.append(audio_file)
        return matches
    
    def find_best_match(self, candidates):
        best_sim = 0
        best_match = None
        for audio_file in candidates:
            wav = preprocess_wav(Path(audio_file))
            embed = self.encoder.embed_utterance(wav)
            similarity = np.dot(self.ref_embed, embed) / (
                np.linalg.norm(self.ref_embed) * np.linalg.norm(embed)
            )
            if similarity > best_sim:
                best_match = audio_file
                best_sim = similarity
        return best_match, best_sim
    
    def find_matches_online(self, candidates):
        matches = []
        for audio_file in candidates:
            wav = preprocess_wav(Path(audio_file))
            embed = self.encoder.embed_utterance(wav)
            similarity = np.dot(self.ref_embed, embed) / (
                np.linalg.norm(self.ref_embed) * np.linalg.norm(embed)
            )
            if similarity > self.threshold:
                matches.append(audio_file)
        return matches



# how to use:

# matcher = SpeakerMatcher(ref_audio_path=r'C:\Users\lahir\code\CREMA-D\AudioWAV\1001_DFA_SAD_XX.wav')

# with files:
# matcher.find_matches(['output_MossFormer2_SS_16K_s1.wav','output_MossFormer2_SS_16K_s2.wav'])

# not using files:

# get reference audio for each speaker
def get_ref_audio():
    data_path = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
    files = os.listdir(data_path)
    files.sort()
    sub = list(set([f.split('_')[0] for f in files]))
    sub.sort()
    paths = {}
    for s in sub:
        first_match = next((f for f in files if f.startswith(s)), None)
        p = os.path.join(data_path, first_match)
        paths[s] = p
    return paths

def process_dir(data_dir, out_dir):
    ref_audio_paths = get_ref_audio()
    os.makedirs("tmp", exist_ok=True)

    for i, file in enumerate(os.listdir(data_dir)):
        print(f'Processing file {i+1}/{len(os.listdir(data_dir))}')
        full_path = os.path.abspath(os.path.join(data_dir, file))
        speaker_sep.process_file(full_path)
        sub = os.path.basename(full_path).split('_')[0]
        ref_path = ref_audio_paths[sub]
        matcher = SpeakerMatcher(ref_audio_path=ref_path, similarity_threshold=0.6)
        tmp_files = [os.path.join('tmp',f) for f in os.listdir("tmp")]
        best_match_path, score = matcher.find_best_match(tmp_files)
        out_path = os.path.join(out_dir, file)
        shutil.copy(best_match_path, out_path)


if __name__ == "__main__":
    process_dir(r'C:\Users\lahir\code\CREMA-D\speech_noise_db15_eval',r'C:\Users\lahir\code\CREMA-D\speech_noise_db15_eval_clean')
    process_dir(r'C:\Users\lahir\code\CREMA-D\speech_noise_db15_train',r'C:\Users\lahir\code\CREMA-D\speech_noise_db15_train_clean')




    


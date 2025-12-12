from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

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


    


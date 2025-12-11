from clearvoice import ClearVoice

class AudioProcessor:
    def __init__(self):
        self.processor = ClearVoice(task='speech_separation',  model_names=['MossFormer2_SS_16K'])
    
    def process_file(self, input_path):
        output = self.processor(input_path=input_path, online_write=False)
        return output
    
'''
how to use:
ap = AudioProcessor()
output = ap.process_file(
    input_path=r'C:\Users\lahir\code\CREMA-D\AudioWAV\1001_DFA_SAD_XX.wav',
)
'''


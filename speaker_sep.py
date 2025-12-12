from clearvoice import ClearVoice

# myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
# output_wav = myClearVoice(input_path=r'C:\Users\lahir\code\CREMA-D\AudioWAV\1001_DFA_SAD_XX.wav', online_write=False)
# myClearVoice.write(output_wav, output_path='output_MossFormer2_SS_16K.wav')

class AudioSeperator:
    def __init__(self):
        self.processor = ClearVoice(task='speech_separation',  model_names=['MossFormer2_SS_16K'])
    
    def process_file(self, input_path, output_path):
        output = self.processor(input_path=input_path, online_write=False)
        self.processor.write(output, output_path=output_path)
        return output
    
# ap = AudioSeperator()
# ap.process_file(
#     input_path=r'C:\Users\lahir\data\noise\temp\noisy_speech.wav',
#     output_path=r'C:\Users\lahir\data\noise\temp\sep\output.wav'
# )

def sep_dataset():
    pass


if __name__ == "__main__":
    sep_dataset()


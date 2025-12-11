from clearvoice import ClearVoice

myClearVoice = ClearVoice(
    task='target_speaker_extraction',
    model_names=['AV_MossFormer2_TSE_16K']  # This model supports audio-only
)

pass
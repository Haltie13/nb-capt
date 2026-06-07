import torch
from torchinfo import summary
from transformers import pipeline
import librosa

device="cuda" if torch.cuda.is_available() else "cpu"

def transcribe(audio_path):
    model_id = "NbAiLab/nb-wav2vec2-1b-bokmaal-v2"
    
    pipe = pipeline(
        "automatic-speech-recognition", 
        model=model_id, 
        device=device
    )

    audio, sr = librosa.load(audio_path, sr=16000)

    result = pipe(audio)
    return result["text"]

if __name__ == "__main__":
    path_to_audio = "data/utale/part_3/group_11/p1_g11_f1_2_t-free.wav"  

    text = transcribe(path_to_audio)
    print(f"Transcription:\n{text}")
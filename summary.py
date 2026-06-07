import torch
from torchinfo import summary
from transformers import pipeline
import librosa

device="cuda" if torch.cuda.is_available() else "cpu"

def print_summary():
    model_id = "NbAiLab/nb-wav2vec2-300m-bokmaal-v2"
    
    pipe = pipeline(
        "automatic-speech-recognition", 
        model=model_id, 
        device=device
    )

    model = pipe.model
    summary(
        model, 
        input_size=(1, 16000), 
        device=device,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=2 
    )


if __name__ == "__main__":
    print(device)

    text = print_summary()
    print(f"Transkrypcja: {text}")
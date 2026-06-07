import json
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from prepare_dataset import processor

class ScorerDataset(Dataset):
    def __init__(self, json_path, datadir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.datadir = Path(datadir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        xml_path = Path(item['audio_path'])
        parts = xml_path.parts
        part_idx = next((i for i, p in enumerate(parts) if p.startswith("part_")), 0)
        audio_path_relative = Path(*parts[part_idx:]).with_suffix('.wav')
        audio_path = self.datadir / audio_path_relative
        
        audio_array, _ = librosa.load(str(audio_path), sr=16000)
        
        input_values = processor(audio_array, sampling_rate=16000).input_values[0]
        
        return {
            'input_values': torch.tensor(input_values),
            'labels': torch.tensor(item['canonical_phonemes'], dtype=torch.long),
            'scores': torch.tensor(item['pseudo_scores'], dtype=torch.float)
        }
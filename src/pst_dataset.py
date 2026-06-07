import json
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from prepare_dataset import processor, vocab_dict

class PSTDataset(Dataset):
    def __init__(self, json_path, cache_path, datadir):
        self.json_path = json_path
        self.cache_path = Path(cache_path)
        self.datadir = Path(datadir)
        self.data = self._load_or_cache()

    def _load_or_cache(self):
        # Invalidate cache if the JSON file has been modified since cache creation
        if self.cache_path.exists() and Path(self.json_path).exists():
            if self.cache_path.stat().st_mtime > Path(self.json_path).stat().st_mtime:
                return torch.load(self.cache_path)
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for sentence_id, recordings in raw_data.items():
            for rec in recordings:
                xml_path = Path(rec['file'])
                parts = xml_path.parts
                part_idx = next((i for i, p in enumerate(parts) if p.startswith("part_")), 0)
                audio_path_relative = Path(*parts[part_idx:]).with_suffix('.wav')
                audio_path = self.datadir / audio_path_relative
                
                audio_array, _ = librosa.load(str(audio_path), sr=16000)
                input_values = processor(audio_array, sampling_rate=16000).input_values[0]
                
                labels = [vocab_dict.get(p['label'], vocab_dict.get("<unk>")) for p in rec['phonemes']]
                
                processed_data.append({
                    'sentence_id': sentence_id,
                    'audio_path': str(rec['file']),
                    'input_values': input_values,
                    'labels': labels
                })
                
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(processed_data, self.cache_path)
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sentence_id': item['sentence_id'],
            'audio_path': item['audio_path'],
            'input_values': torch.tensor(item['input_values']),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }
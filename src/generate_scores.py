import torch
import json
import torch.nn.functional as F
from collections import defaultdict
from safetensors.torch import load_file
from apr import FullAPRModel
from pst_dataset import PSTDataset
from prepare_dataset import vocab_dict

def extract_representations(model, input_values, labels, device):
    input_values = input_values.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    
    with torch.no_grad():
        ssl_features = model.wav2vec2(input_values).last_hidden_state
        start_tokens = torch.full((labels.size(0), 1), model.start_token_id, dtype=torch.long, device=device)
        decoder_inputs = torch.cat([start_tokens, labels], dim=1)
        _, dec_r, _ = model.pprm(ssl_features, decoder_inputs)
        
    return dec_r.squeeze(0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FullAPRModel(
        model_name="NbAiLab/nb-wav2vec2-300m-bokmaal-v2",
        vocab_size=len(vocab_dict),
        pad_token_id=vocab_dict.get("<pad>", 0),
        start_token_id=vocab_dict.get("<start>", 1),
        end_token_id=vocab_dict.get("<end>", 2)
    )
    
    state_dict = load_file("checkpoints/e2er-apr-stage/checkpoint-4500/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    native_dataset = PSTDataset("data/pst/native_oslo.json", "data/cache/native_oslo.pt", "data/utale")
    l2_dataset = PSTDataset("data/pst/l2_speakers.json", "data/cache/l2_speakers.pt", "data/utale")

    canonical_labels_dict = {}
    for i in range(len(native_dataset)):
        item = native_dataset[i]
        sid = item['sentence_id']
        if sid not in canonical_labels_dict:
            canonical_labels_dict[sid] = item['labels']

    native_representations = defaultdict(list)
    
    for i in range(len(native_dataset)):
        item = native_dataset[i]
        sid = item['sentence_id']
        canonical_labels = canonical_labels_dict[sid]
        
        r = extract_representations(model, item['input_values'], canonical_labels, device)
        native_representations[sid].append(r)

    native_centroids = {}
    for sid, reps in native_representations.items():
        stacked = torch.stack(reps)
        native_centroids[sid] = stacked.mean(dim=0)

    results = []

    for ds, is_native in [(native_dataset, True), (l2_dataset, False)]:
        for i in range(len(ds)):
            item = ds[i]
            sid = item['sentence_id']
            
            if sid not in native_centroids:
                continue
            
            canonical_labels = canonical_labels_dict[sid]
                
            r = extract_representations(model, item['input_values'], canonical_labels, device)
            centroid = native_centroids[sid]
            
            sim = F.cosine_similarity(r, centroid, dim=-1)
            scores = torch.clamp(sim, min=0.0).cpu().tolist()
            
            results.append({
                "audio_path": item['audio_path'],
                "is_native": is_native,
                "sentence_id": sid,
                "canonical_phonemes": canonical_labels.tolist(),
                "pseudo_scores": scores
            })

    with open("data/pst_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
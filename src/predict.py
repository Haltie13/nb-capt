import numpy as np
import phonecodes.phonecodes as pc
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import Wav2Vec2ForCTC
from apr import FullAPRModel
from prepare_dataset import get_or_create_processed_dataset, processor, datadir, vocab_dict

def to_ipa(text):
    return ' '.join(w if w.startswith('<') and w.endswith('>') else pc.xsampa2ipa(w) for w in text.split())

def main():
    processed_dataset = get_or_create_processed_dataset()

    sample_count = 10
    rng = np.random.default_rng(42)
    sample_count = min(sample_count, len(processed_dataset))
    sample_indices = rng.choice(len(processed_dataset), size=sample_count, replace=False)
    eval_dataset = processed_dataset.select(sample_indices)

    checkpoints_dir = datadir.parent / 'checkpoints'
    model_paths = [
        checkpoints_dir / 'e2er-apr-stage' / 'checkpoint-4500', 
        checkpoints_dir / 'wav2vec2-300m-phoneme-ctc' / 'checkpoint-2500',
    ]

    label_str = processor.batch_decode(eval_dataset["labels"], group_tokens=False)

    all_results = []

    for model_path in model_paths:
        is_apr = "e2er-apr-stage" in str(model_path)
        if is_apr:
            model = FullAPRModel(
                model_name="NbAiLab/nb-wav2vec2-300m-bokmaal-v2",
                vocab_size=len(vocab_dict),
                pad_token_id=vocab_dict.get("<pad>", 0),
                start_token_id=vocab_dict.get("<start>", 1),
                end_token_id=vocab_dict.get("<end>", 2),
            )
            state_dict = load_file(str(Path(model_path) / "model.safetensors"))
            model.load_state_dict(state_dict)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(
                str(model_path),
                ignore_mismatched_sizes=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        pred_ids_list = []
        for i in range(len(eval_dataset)):
            input_values = torch.tensor(eval_dataset[i]["input_values"]).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_apr:
                    outputs = model(input_values)
                    # APR returns (ctc_logits, decoder_out)
                    logits = outputs[0]
                else:
                    outputs = model(input_values)
                    logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
                pred_ids_list.append(pred_ids)

        pred_str = processor.batch_decode(pred_ids_list)

        all_results.append({
            "model_path": model_path,
            "pred_str": pred_str,
        })

        del model
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    for i in range(len(eval_dataset)):
        label_ipa = to_ipa(label_str[i])
        print(f"File: {eval_dataset[i]['audio_path']}")
        print(f"Label:  [{label_ipa}]")
        for j, result in enumerate(all_results):
            pred_ipa = to_ipa(result['pred_str'][i])
            print(f"Model {j+1}: [{pred_ipa}]")
        print("-" * 50)

if __name__ == "__main__":
    main()

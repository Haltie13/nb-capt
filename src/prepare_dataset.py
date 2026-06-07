from datasets import Dataset, load_from_disk
from pathlib import Path
from parse_transcription import parse_transcription
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import json

rootdir = Path(__file__).parents[1]
datadir = rootdir / 'data'

vocab_path = rootdir / 'tmp' / 'vocab.json'
with open(vocab_path, "r") as f:
    vocab_dict = json.load(f)

tokenizer = Wav2Vec2CTCTokenizer(
    str(vocab_path),
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal-v2")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def create_raw_dataset():
    part1_trans = datadir / 'Annotation' / 'part_1.trans'
    part2_trans = datadir / 'Annotation' / 'part_2.trans'
    parts = [part1_trans, part2_trans]

    parsed_data = {}
    for part in parts:
        sub_pd = parse_transcription(part)
        parsed_data.update(sub_pd)

    audio_paths = list(parsed_data.keys())
    token_sequences = list(parsed_data.values())

    raw_dataset = Dataset.from_dict({
        "audio_path": audio_paths,
        "tokens": token_sequences
    })

    return raw_dataset

def prepare_dataset(batch):
    audio_path = str(datadir / 'utale' / batch["audio_path"])
    audio_array, _ = librosa.load(audio_path, sr=16000)
    
    batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]

    batch["labels"] = [vocab_dict.get(token, 1) for token in batch["tokens"]]
    
    return batch

def get_or_create_processed_dataset():
    cache_path = datadir / 'cache'
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(str(cache_path))
    
    print("Cached dataset not found. Generating.")
    raw_dataset = create_raw_dataset()
    filtered_dataset = raw_dataset.filter(lambda x: librosa.get_duration(path=str(datadir / 'utale' / x["audio_path"])) <= 15.0)
    prepared_dataset = filtered_dataset.map(prepare_dataset, remove_columns=["tokens"], num_proc=1)
    
    prepared_dataset.save_to_disk(str(cache_path))
    return prepared_dataset


def main():
    raw_dataset = create_raw_dataset()

    over_15s = sum(1 for item in raw_dataset if librosa.get_duration(path=str(datadir / 'utale' / item["audio_path"])) > 15.0)
    count = sum(1 for item in raw_dataset)
    max_dur = max(librosa.get_duration(path=str(datadir / 'utale' / item["audio_path"])) for item in raw_dataset if librosa.get_duration(path=str(datadir / 'utale' / item["audio_path"])))
    total_dur = sum(librosa.get_duration(path=str(datadir / 'utale' / item["audio_path"])) for item in raw_dataset if librosa.get_duration(path=str(datadir / 'utale' / item["audio_path"])))
    print(f"Number of audio files over 15 seconds: {over_15s}")
    print(f'Number of all recordings: {count}')
    print(f'Maximal duration: {max_dur}')
    print(f'Total duration: {total_dur}')
    # prepared_dataset = raw_dataset.map(prepare_dataset, remove_columns=["audio_path", "tokens"], num_proc=1)

if __name__ == "__main__":
    main()
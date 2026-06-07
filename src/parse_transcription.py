import re

AUDIO_FORMAT = 'wav'


modifiers = ['""', '"', '%', '_=', ':']
specials = ["<start>", "<end>", "<sil>", "<fp>", "<inhale>", "<exhale>", "<vowel>", "<nasal>"]

pattern = re.compile(r'(' + '|'.join(re.escape(s) for s in (specials + modifiers)) + r')')

def tokenize_phoneme_string(text):
    raw_splits = re.split(pattern, text)
    return [t for t in raw_splits if t != '']

def parse_transcription(file_path):
    dataset_dict = {}
    current_audio = None
    current_tokens = []
    current_word = None

    with open(file_path, 'r', encoding='utf-8') as f:
        next(f, None) 
        
        for line in f:
            line = line.strip()
            if not line: 
                continue

            if line == ".":
                if current_audio and current_tokens:
                    dataset_dict[current_audio] = current_tokens
                    current_audio = None
                    current_tokens = []
                    current_word = None
                continue

            if line.startswith('"'):
                if current_audio and current_tokens:
                    dataset_dict[current_audio] = current_tokens

                current_audio = line.replace('"', '') + f'.{AUDIO_FORMAT}'
                current_tokens = []
                current_word = None
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                raw_token = parts[2]
                expanded_tokens = tokenize_phoneme_string(raw_token) 
                
                if len(parts) >= 4:
                    word = parts[3]
                    if current_word is not None and word != current_word:
                        current_tokens.append('|')
                    current_word = word
                
                current_tokens.extend(expanded_tokens)
            
    if current_audio and current_tokens:
        dataset_dict[current_audio] = current_tokens
        
    return dataset_dict

if __name__ == "__main__":
    from pathlib import Path
    from itertools import islice

    data_dir = Path(__file__).parents[1] / 'data'
    part1_trans = data_dir / 'Annotation' / 'part_1.trans'
    if not part1_trans.exists():
        print(f'File doesnt exist: {part1_trans.resolve()}')
        exit()
    dataset = parse_transcription(part1_trans)

    top5 = islice(dataset.items(), 5)
    for key, value in top5:
        print(key, value)
from pathlib import Path
import json

# Update to include all parts if needed
files = [
    Path("data/Annotation/part_2.trans"),
    Path("data/Annotation/part_1.trans"),
    # Path("data/Annotation/part_3.trans"),
]

special_tokens = {"<start>", "<end>", "<sil>", "<fp>", "<inhale>", "<exhale>", "<vowel>", "<nasal>", '|'}
modifiers = {'"', '""', '%', '_=', ':'}

total = 0
vocab = set()
base_tokens = set()
special_found = set()
unique = set()

for fp in files:
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "." or line.startswith("#!") or line.startswith('"'):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            token = parts[2]
            unique.add(token)

            base = token
            for p in modifiers:
                base = base.replace(p, '')
            base_tokens.add(base)

            total += 1
            if token.startswith("<") and token.endswith(">"):
                special_found.add(token)
            

vocab = sorted(base_tokens | special_found | modifiers | {'|'})

extra_specials = ["<pad>", "<unk>"]
vocab = extra_specials + vocab

vocab_json = {tok: i for i, tok in enumerate(vocab)}

print("Total tokens:", total)
print("Unique tokens:", len(unique))
print(unique)
# print("Special tokens:")
# print(special_found)
print("Base tokens:", len(base_tokens))
print(sorted(base_tokens))


Path("tmp/vocab.json").write_text(json.dumps(vocab_json, ensure_ascii=True, indent=2))
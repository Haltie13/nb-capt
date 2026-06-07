import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json

ANNOTATION_DIR = "./data/Annotation"
NATIVE_GROUPS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 24]
L2_GROUPS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
NATIVE_OUTPUT = "./data/pst/native_oslo.json"
L2_OUTPUT = "./data/pst/l2_speakers.json"
MERGED_OUTPUT = "./data/pst/merged.json"

def extract_phonemes_from_node(annotation_node):
    ignore_tags = {'<start>', '<end>', '<sil>', '<inhale>', '<exhale>', '<fp>', '<nasal>', '<vowel>'}
    pure_phonemes = []
    
    for word in annotation_node.findall('word'):
        word_text = word.get('text')
        if word_text in ignore_tags:
            continue
            
        for seg in word.findall('seg'):
            label = seg.get('label')
            if label not in ignore_tags:
                pure_phonemes.append({
                    'label': label,
                    'start': float(seg.get('start')),
                    'end': float(seg.get('end'))
                })
                
    return pure_phonemes

def build_dataset_for_groups(annotation_dir, group_ids):
    dataset = defaultdict(list)
    
    parts = ["1", "2"]
    
    group_prefixes = {}
    for part in parts:
        for g in group_ids:
            group_str = f"group_{int(g):02d}"
            prefix = f"part_{part}/{group_str}/"
            group_prefixes[prefix] = (part, group_str)
            
    for part in parts:
        xml_path = Path(annotation_dir) / f"part_{part}.xml"
        if not xml_path.exists():
            continue
            
        print(f"Parsing {xml_path}...")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for annotation_node in root.findall('annotation'):
            full_id = annotation_node.get('id')
            
            matched_prefix = None
            for prefix in group_prefixes:
                if prefix in full_id:
                    matched_prefix = prefix
                    break
            if not matched_prefix:
                continue
                
            sentence_id = full_id.split('-')[-1]
            part_str, group_str = group_prefixes[matched_prefix]
            
            phonemes = extract_phonemes_from_node(annotation_node)
            
            inferred_xml_file = Path(annotation_dir) / f"part_{part_str}" / group_str / f"{full_id.split('/')[-1]}.xml"
            dataset[sentence_id].append({
                'file': str(inferred_xml_file),
                'phonemes': phonemes
            })
            
    return dataset

def filter_common_texts(native_data, l2_data):
    common_keys = set(native_data.keys()) & set(l2_data.keys())
    print(f"Found {len(common_keys)} common texts (sentence IDs).")
    
    filtered_native = {k: native_data[k] for k in common_keys}
    filtered_l2 = {k: l2_data[k] for k in common_keys}
    
    return filtered_native, filtered_l2

def merge_datasets(native_data, l2_data):
    merged_data = {}
    for k in native_data:
        merged_data[k] = native_data[k] + l2_data.get(k, [])
    return merged_data

def export_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"Saved dataset to {output_path}")

def main():
    print("Building native dataset...")
    native_data = build_dataset_for_groups(ANNOTATION_DIR, NATIVE_GROUPS)
    
    print("Building L2 dataset...")
    l2_data = build_dataset_for_groups(ANNOTATION_DIR, L2_GROUPS)
    
    print("Filtering common texts...")
    filtered_native, filtered_l2 = filter_common_texts(native_data, l2_data)
    
    print("Merging datasets...")
    merged_data = merge_datasets(filtered_native, filtered_l2)
    
    print("Saving datasets...")
    export_dataset(filtered_native, NATIVE_OUTPUT)
    export_dataset(filtered_l2, L2_OUTPUT)
    export_dataset(merged_data, MERGED_OUTPUT)

if __name__ == "__main__":
    main()
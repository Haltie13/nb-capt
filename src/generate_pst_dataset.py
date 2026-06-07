import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json
import argparse

def get_xml_files_for_group(annotation_dir, group_id, part="1"):
    group_str = f"group_{int(group_id):02d}"
    path = Path(annotation_dir) / f"part_{part}" / group_str
    return list(path.rglob("*.xml"))

def extract_pure_phonemes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation_node = root.find('annotation')
    
    full_id = annotation_node.get('id')
    sentence_id = full_id.split('-')[-1]
    
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
                
    return sentence_id, pure_phonemes

def build_regional_dataset(annotation_dir, group_id, part="1"):
    group_str = f"group_{int(group_id):02d}"
    group_dir = Path(annotation_dir) / f"part_{part}" / group_str
    
    dataset = defaultdict(list)
    
    if group_dir.exists() and list(group_dir.rglob("*.xml")):
        xml_files = list(group_dir.rglob("*.xml"))
        for xml_file in xml_files:
            sentence_id, phonemes = extract_pure_phonemes(xml_file)
            dataset[sentence_id].append({
                'file': str(xml_file),
                'phonemes': phonemes
            })
    else:
        big_xml_path = Path(annotation_dir) / f"part_{part}.xml"
        if big_xml_path.exists():
            tree = ET.parse(big_xml_path)
            root = tree.getroot()
            
            group_prefix = f"part_{part}/{group_str}/"
            ignore_tags = {'<start>', '<end>', '<sil>', '<inhale>', '<exhale>', '<fp>', '<nasal>', '<vowel>'}
            
            for annotation_node in root.findall('annotation'):
                full_id = annotation_node.get('id')
                if group_prefix not in full_id:
                    continue
                    
                sentence_id = full_id.split('-')[-1]
                
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
                
                inferred_xml_file = Path(annotation_dir) / f"part_{part}" / group_str / f"{full_id.split('/')[-1]}.xml"
                dataset[sentence_id].append({
                    'file': str(inferred_xml_file),
                    'phonemes': pure_phonemes
                })
        else:
            print(f"Warning: Neither the directory '{group_dir}' nor the file '{big_xml_path}' exists.")
            
    return dataset

def export_dataset(dataset, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--part", type=str, default="1")
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    regional_data = build_regional_dataset(args.annotation_dir, args.group_id, args.part)
    export_dataset(regional_data, args.output)

if __name__ == "__main__":
    main()
import os
import re
import json
from tqdm import tqdm

NAME_MAP_FILE = "data/1_id_to_name.txt"
SCHEMATIC_DIR = "data/2_named_schematics"
OUTPUT_JSON = "data/5_labels.json"

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text).strip('_')
    return text

if __name__ == "__main__":
    print(f"--- Generating Labels File: '{OUTPUT_JSON}' ---")

    try:
        final_filenames = set(os.listdir(SCHEMATIC_DIR))
    except FileNotFoundError:
        print(f"ERROR: Directory not found: '{SCHEMATIC_DIR}'. Please ensure previous steps have been run.")
        exit()
    
    print(f"Found {len(final_filenames)} files in '{SCHEMATIC_DIR}'.")
    labels = {}

    print(f"Reading name map from '{NAME_MAP_FILE}' to create labels...")
    try:
        with open(NAME_MAP_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Matching names to files"):
                if ' - ' in line:
                    parts = line.strip().split(' - ', 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        id_str, original_name = parts
                        
                        slug = slugify(original_name)
                        
                        expected_schem = f"{slug}_{id_str}.schem"
                        expected_schematic = f"{slug}_{id_str}.schematic"

                        if expected_schem in final_filenames:
                            labels[expected_schem] = original_name
                        elif expected_schematic in final_filenames:
                            labels[expected_schematic] = original_name

    except FileNotFoundError:
        print(f"FATAL ERROR: Name map file not found at '{NAME_MAP_FILE}'")
        exit()

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=4)
        
    print("\n--- Label Generation Complete ---")
    print(f"Successfully created '{OUTPUT_JSON}' with {len(labels)} entries.")
    print("This file will be used by the training script to get text prompts.")
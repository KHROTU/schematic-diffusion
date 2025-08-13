import os
import re
import shutil
from tqdm import tqdm

RAW_DIR = "data/0_raw_downloads"
NAME_MAP_FILE = "data/1_id_to_name.txt"
OUTPUT_DIR = "data/2_named_schematics"

def slugify(text):
    """
    Converts a string into a safe, lowercase filename slug.
    Example: "The Tavern" -> "the_tavern"
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text).strip('_')
    return text

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Reading name map from '{NAME_MAP_FILE}'...")
    id_to_name = {}
    try:
        with open(NAME_MAP_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if ' - ' in line:
                    parts = line.strip().split(' - ', 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        id_str, name = parts
                        id_to_name[id_str] = name
    except FileNotFoundError:
        print(f"FATAL ERROR: Name map file not found at '{NAME_MAP_FILE}'")
        exit()
    
    print(f"Loaded {len(id_to_name)} ID-to-name mappings.")

    print(f"Processing files from '{RAW_DIR}'...")
    raw_files = os.listdir(RAW_DIR)
    renamed_count = 0
    unmapped_count = 0

    for filename in tqdm(raw_files, desc="Renaming files"):
        file_id = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]

        if file_id in id_to_name:
            original_name = id_to_name[file_id]
            new_slug = slugify(original_name)
            
            new_filename = f"{new_slug}_{file_id}{file_ext}"
            
            source_path = os.path.join(RAW_DIR, filename)
            dest_path = os.path.join(OUTPUT_DIR, new_filename)
            
            shutil.copy(source_path, dest_path)
            renamed_count += 1
        else:
            unmapped_count += 1

    print("\n--- Renaming Complete ---")
    print(f"Successfully renamed and copied: {renamed_count} files.")
    print(f"Files with no matching ID in map: {unmapped_count} (these were skipped).")
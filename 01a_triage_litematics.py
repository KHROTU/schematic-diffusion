import os
import shutil
from tqdm import tqdm

SOURCE_DIR = "data/2_named_schematics"

DEST_DIR = "data/3_litematics_to_convert"

if __name__ == "__main__":
    print("--- Litematic Triage Stage ---")
    
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory '{SOURCE_DIR}' not found.")
        print("Please run '01_rename_files.py' first to create and populate it.")
        exit()
    
    os.makedirs(DEST_DIR, exist_ok=True)
    
    all_files = os.listdir(SOURCE_DIR)
    litematic_files = [f for f in all_files if f.lower().endswith('.litematic')]
    
    if not litematic_files:
        print("No .litematic files found in the source directory. Nothing to do.")
        exit()
        
    print(f"Found {len(litematic_files)} .litematic files to move for conversion.")
    
    for filename in tqdm(litematic_files, desc="Moving litematics"):
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)
        
        try:
            shutil.move(source_path, dest_path)
        except Exception as e:
            print(f"\nCould not move file {filename}. Error: {e}")
            
    print("\n--- Triage Complete ---")
    print(f"All {len(litematic_files)} litematic files have been moved to '{DEST_DIR}'.")
    print("You can now proceed with running the litematic converter.")
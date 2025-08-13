import torch
from nbt import nbt
import os
import numpy as np
import json
from tqdm import tqdm

SCHEMATIC_SIZE = 32 
MAX_INPUT_DIM = 256 

INPUT_SCHEMATIC_DIR = "data/2_named_schematics"
OUTPUT_TENSOR_DIR = "data/4_processed_tensors"
LABELS_PATH = "data/5_labels.json"

BLOCK_VOCAB = {
    'minecraft:air': 0, 'minecraft:stone': 1, 'minecraft:dirt': 2, 'minecraft:grass_block': 3,
    'minecraft:sand': 4, 'minecraft:gravel': 5, 'minecraft:water': 6, 'minecraft:lava': 7,
    'minecraft:obsidian': 8, 'generic_log': 9, 'generic_planks': 10, 'generic_slab': 11,
    'generic_stairs': 12, 'generic_fence': 13, 'generic_trapdoor': 14, 'minecraft:cobblestone': 15,
    'minecraft:stone_bricks': 16, 'minecraft:sandstone': 17, 'minecraft:netherrack': 18,
    'minecraft:nether_bricks': 19, 'minecraft:end_stone': 20, 'minecraft:end_stone_bricks': 21,
    'minecraft:quartz_block': 22, 'minecraft:blackstone': 23, 'generic_stone_slab': 24,
    'generic_stone_stairs': 25, 'generic_stone_wall': 26, 'minecraft:bricks': 27,
    'minecraft:glass': 28, 'minecraft:glass_pane': 29, 'minecraft:iron_bars': 30,
    'minecraft:white_wool': 31, 'minecraft:black_wool': 32, 'minecraft:gray_wool': 33,
    'minecraft:light_gray_wool': 34, 'minecraft:red_wool': 35, 'minecraft:blue_wool': 36,
    'minecraft:brown_wool': 37, 'minecraft:green_wool': 38, 'minecraft:white_terracotta': 39,
    'minecraft:black_terracotta': 40, 'minecraft:gray_terracotta': 41, 'minecraft:red_terracotta': 42,
    'minecraft:blue_terracotta': 43, 'minecraft:brown_terracotta': 44, 'minecraft:white_concrete': 45,
    'minecraft:black_concrete': 46, 'minecraft:gray_concrete': 47, 'minecraft:glowstone': 48,
    'minecraft:sea_lantern': 49, 'minecraft:shroomlight': 50, 'minecraft:lantern': 51,
    'minecraft:torch': 52, 'minecraft:bookshelf': 53, 'minecraft:ladder': 54,
    'minecraft:chain': 55, 'minecraft:leaves': 56
}
NUM_BLOCK_TYPES = len(BLOCK_VOCAB)

def get_clean_block_name(block_state_str):
    return str(block_state_str).replace('minecraft:', '').split('[')[0]

def map_name_to_vocab(clean_name):
    full_name = f'minecraft:{clean_name}'
    if full_name in BLOCK_VOCAB: return BLOCK_VOCAB[full_name]
    if any(s in clean_name for s in ['_log', '_stem', '_wood', 'hyphae']): return BLOCK_VOCAB['generic_log']
    if '_planks' in clean_name: return BLOCK_VOCAB['generic_planks']
    if '_slab' in clean_name:
        return BLOCK_VOCAB['generic_stone_slab'] if any(s in clean_name for s in ['stone', 'brick', 'cobble', 'sandstone', 'quartz', 'blackstone', 'purpur', 'nether', 'prismarine', 'andesite', 'diorite', 'granite']) else BLOCK_VOCAB['generic_slab']
    if '_stairs' in clean_name:
        return BLOCK_VOCAB['generic_stone_stairs'] if any(s in clean_name for s in ['stone', 'brick', 'cobble', 'sandstone', 'quartz', 'blackstone', 'purpur', 'nether', 'prismarine', 'andesite', 'diorite', 'granite']) else BLOCK_VOCAB['generic_stairs']
    if '_fence' in clean_name and '_gate' not in clean_name: return BLOCK_VOCAB['generic_fence']
    if '_trapdoor' in clean_name: return BLOCK_VOCAB['generic_trapdoor']
    if '_wall' in clean_name: return BLOCK_VOCAB['generic_stone_wall']
    if '_leaves' in clean_name: return BLOCK_VOCAB['minecraft:leaves']
    return BLOCK_VOCAB['minecraft:air']

def process_schematic_file(filepath):
    try:
        nbt_file = nbt.NBTFile(filepath, 'rb')
    except Exception:
        return None

    if 'Width' in nbt_file and 'Height' in nbt_file and 'Length' in nbt_file:
        W, H, L = nbt_file['Width'].value, nbt_file['Height'].value, nbt_file['Length'].value
        voxels = np.zeros((H, L, W), dtype=np.int64)

        if 'Blocks' in nbt_file:
            block_ids = np.array(nbt_file['Blocks'].value).reshape(H, L, W)
            LEGACY_ID_MAP = {1: 1, 2: 3, 3: 2, 4: 15, 5: 10, 17: 9, 20: 28, 44: 24, 45: 27, 49: 8, 89: 48, 98: 16, 121: 20}
            for old_id, new_id in LEGACY_ID_MAP.items(): voxels[block_ids == old_id] = new_id
        
        elif 'BlockData' in nbt_file and 'Palette' in nbt_file:
            palette_tag = nbt_file['Palette']
            block_data_indices = np.array(nbt_file['BlockData'].value)
            
            expected_size = W * H * L
            if len(block_data_indices) != expected_size:
                return None # skip broken files
            
            max_idx = max(tag.value for tag in palette_tag.values())
            palette_map = np.zeros(max_idx + 1, dtype=np.int64)
            for state_str, idx_tag in palette_tag.items():
                palette_map[idx_tag.value] = map_name_to_vocab(get_clean_block_name(state_str))
            
            voxels = palette_map[block_data_indices.reshape(H, L, W)]
        else: return None

    elif 'size' in nbt_file and 'palette' in nbt_file and 'blocks' in nbt_file:
        size = nbt_file['size']
        W, H, L = size[0].value, size[1].value, size[2].value
        palette_tag = nbt_file['palette']
        palette_map = np.zeros(len(palette_tag), dtype=np.int64)
        for i, block_type in enumerate(palette_tag):
            palette_map[i] = map_name_to_vocab(get_clean_block_name(block_type['Name'].value))
        
        voxels = np.zeros((H, L, W), dtype=np.int64)
        for block in nbt_file['blocks']:
            pos = block['pos']
            x, y, z = pos[0].value, pos[1].value, pos[2].value
            if 0 <= x < W and 0 <= y < H and 0 <= z < L:
                voxels[y, z, x] = palette_map[block['state'].value]

    else: return None

    if not (0 < W < MAX_INPUT_DIM and 0 < H < MAX_INPUT_DIM and 0 < L < MAX_INPUT_DIM):
        return None

    padded_voxels = np.zeros((SCHEMATIC_SIZE, SCHEMATIC_SIZE, SCHEMATIC_SIZE), dtype=np.int64)
    h, d, w = voxels.shape
    copy_h, copy_d, copy_w = min(h, SCHEMATIC_SIZE), min(d, SCHEMATIC_SIZE), min(w, SCHEMATIC_SIZE)
    padded_voxels[:copy_h, :copy_d, :copy_w] = voxels[:copy_h, :copy_d, :copy_w]

    voxels_torch = torch.from_numpy(padded_voxels)
    one_hot = torch.nn.functional.one_hot(voxels_torch, num_classes=NUM_BLOCK_TYPES)
    return one_hot.permute(3, 1, 0, 2).float()

if __name__ == '__main__':
    os.makedirs(OUTPUT_TENSOR_DIR, exist_ok=True)
    
    print(f"Loading labels from '{LABELS_PATH}'...")
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Labels file not found at '{LABELS_PATH}'.")
        print("Please run '02_generate_labels.py' to create it.")
        exit()

    files_to_process = list(labels_data.keys())
    processed_count, skipped_count = 0, 0

    print(f"\n--- Starting Final Preprocessing ---")
    print(f"Using 'NBT' library with a {NUM_BLOCK_TYPES}-block vocabulary.")
    print(f"Found {len(files_to_process)} labeled schematics to process.")
    print(f"Output tensors will be saved to '{OUTPUT_TENSOR_DIR}'")
    
    for filename in tqdm(files_to_process, desc="Processing Schematics"):
        input_path = os.path.join(INPUT_SCHEMATIC_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".pt"
        output_path = os.path.join(OUTPUT_TENSOR_DIR, output_filename)

        if os.path.exists(output_path):
            processed_count += 1
            continue
        
        tensor = process_schematic_file(input_path)
        
        if tensor is not None:
            torch.save(tensor, output_path)
            processed_count += 1
        else:
            skipped_count += 1
            
    print("\n--- Preprocessing Complete ---")
    print(f"Successfully processed and saved: {processed_count} tensors.")
    print(f"Skipped (corrupt, too large, or format error): {skipped_count} schematics.")
    print(f"Your training-ready tensors are in: '{OUTPUT_TENSOR_DIR}'")
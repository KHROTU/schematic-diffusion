import torch
import numpy as np
from nbt import nbt
import os
import clip
from train_diffusion import UNet3D, DiffusionScheduler
from config import BLOCK_VOCAB, NUM_BLOCK_TYPES
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEMATIC_SIZE = 32

PROMPT = "a small tower made of stone bricks and generic logs"
MODEL_PATH = "models/schematic_diffusion_epoch_95.pth" 
GUIDANCE_SCALE = 7.5
INFERENCE_STEPS = 250
OUTPUT_FILENAME = "generated_tower.schem"
OUTPUT_DIR = "output"

print("Loading models...")
text_embed_dim = 512 
model = UNet3D(in_c=NUM_BLOCK_TYPES, txt_d=text_embed_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval() 

clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
scheduler = DiffusionScheduler(steps=1000, device=DEVICE)
print("Models loaded successfully.")

def get_text_embedding(prompts, device=DEVICE):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
    return text_features.float()

def save_to_schematic(block_indices_np, filename):
    INV_BLOCK_VOCAB = {v: k for k, v in BLOCK_VOCAB.items()}
    D, H, W = block_indices_np.shape
    
    schem = nbt.NBTFile()
    schem.name = "Root"
    schem.tags.extend([
        nbt.TAG_Short(name="Width", value=W), nbt.TAG_Short(name="Height", value=H), nbt.TAG_Short(name="Length", value=D),
        nbt.TAG_Int(name="PaletteMax", value=0), nbt.TAG_Compound(name="Palette"), nbt.TAG_Byte_Array(name="BlockData"),
        nbt.TAG_List(name="Entities", type=nbt.TAG_Compound), nbt.TAG_List(name="TileEntities", type=nbt.TAG_Compound)
    ])

    schem_palette, vocab_to_palette_map, p_idx = {}, {}, 0
    unique_indices = np.unique(block_indices_np)
    if 0 not in unique_indices: unique_indices = np.insert(unique_indices, 0, 0)

    for v_idx in unique_indices:
        b_name = INV_BLOCK_VOCAB.get(int(v_idx), "minecraft:air")
        if "generic" in b_name:
            if "log" in b_name: b_name = "minecraft:oak_log"
            elif "planks" in b_name: b_name = "minecraft:oak_planks"
            else: b_name = "minecraft:air" 
        
        if b_name not in schem_palette:
            schem_palette[b_name], vocab_to_palette_map[v_idx], p_idx = p_idx, p_idx, p_idx + 1
            schem["Palette"].tags.append(nbt.TAG_Int(name=b_name, value=schem_palette[b_name]))

    schem["PaletteMax"].value = len(schem_palette)
    palette_indices = np.array([vocab_to_palette_map.get(idx, 0) for idx in block_indices_np.flatten()], dtype=np.uint8)
    schem["BlockData"].value = bytearray(palette_indices)
    
    schem.write_file(filename)
    print(f"\nSchematic saved to {filename}")

@torch.no_grad()
def generate(prompt, model, scheduler, guidance_scale, steps, shape):
    cond_embedding = get_text_embedding([prompt])
    uncond_embedding = get_text_embedding([""]) 
    latents = torch.randn(shape, device=DEVICE)

    for t in tqdm(reversed(range(0, scheduler.steps, scheduler.steps // steps)), desc="Generating"):
        timestep = torch.tensor([t] * shape[0], device=DEVICE)
        pred_noise_cond = model(latents, timestep, cond_embedding)
        pred_noise_uncond = model(latents, timestep, uncond_embedding)
        noise_pred = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        alpha_t, alpha_cumprod_t, beta_t = scheduler.alphas[t], scheduler.alpha_cumprod[t], scheduler.betas[t]
        latents = (1/torch.sqrt(alpha_t)) * (latents - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t)) * noise_pred)
        if t > 0: latents += torch.sqrt(beta_t) * torch.randn_like(latents)
    return latents

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    shape = (1, NUM_BLOCK_TYPES, SCHEMATIC_SIZE, SCHEMATIC_SIZE, SCHEMATIC_SIZE)
    final_latents = generate(PROMPT, model, scheduler, GUIDANCE_SCALE, INFERENCE_STEPS, shape)
    
    final_latents = (final_latents + 1.0) / 2.0
    final_latents = final_latents.clamp(0, 1)
    
    final_block_indices = torch.argmax(final_latents, dim=1).squeeze(0)
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    save_to_schematic(final_block_indices.cpu().numpy(), full_output_path)
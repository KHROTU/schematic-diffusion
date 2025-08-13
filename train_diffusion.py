import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import clip
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DIR = "data/4_processed_tensors"
LABELS_PATH = "data/5_labels.json"
SCHEMATIC_SIZE = 32
NUM_BLOCK_TYPES = 57
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5

# Recommendations by ChatGPT, how much to trust is up to you.
# For me (RTX 4070 laptop GPU), 8:20 per epoch, 14 hrs total.
#
# | Scenario | Key Change | Est. VRAM Usage | Est. Time / Epoch | Total Time (100 Epochs) |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | **Baseline** | `batch=12`, `size=32`, `base_c=64` | **~4-6 GB** | **~10-20 min** | **~16-33 hours** |
# | **Larger Batch** | `batch=24` | **~7-10 GB** | **~6-12 min** | **~10-20 hours** |
# | **Larger Schematics** | `size=64` | **> 24 GB (OOM)** | **~1.5-3 hours** | **~6-12 days** |
# | **Wider Model** | `base_c=128` | **~12-18 GB** | **~30-50 min** | **~50-83 hours** |
# | **Mixed Precision** | `torch.amp` | **~3-4 GB** | **~6-12 min** | **~10-20 hours** |

class PreprocessedSchematicDataset(Dataset):
    def __init__(self, tensor_dir, labels_path):
        self.tensor_dir = tensor_dir
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
            
        self.tensor_files = []
        for schem_filename in self.labels.keys():
            base_name = os.path.splitext(schem_filename)[0]
            tensor_path = os.path.join(self.tensor_dir, f"{base_name}.pt")
            if os.path.exists(tensor_path):
                self.tensor_files.append(base_name)
        print(f"Found {len(self.tensor_files)} processed tensors with labels.")

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        base_name = self.tensor_files[idx]
        original_filename = ""
        for ext in ['.schem', '.schematic', '.nbt']:
            if f"{base_name}{ext}" in self.labels:
                original_filename = f"{base_name}{ext}"
                break
        
        text_prompt = self.labels[original_filename]
        schematic_tensor = torch.load(os.path.join(self.tensor_dir, f"{base_name}.pt"), weights_only=True)
        return schematic_tensor, text_prompt

def get_text_embedding(prompts, clip_model_local, device=DEVICE):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        text_features = clip_model_local.encode_text(text_tokens)
    return text_features.float()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time):
        device = time.device; half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_c, out_c, time_dim, drop):
        super().__init__(); self.conv1 = nn.Conv3d(in_c, out_c, 3, 1, 1); self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1); self.relu = nn.ReLU(); self.norm1 = nn.BatchNorm3d(out_c); self.norm2 = nn.BatchNorm3d(out_c); self.drop = nn.Dropout(drop); self.time_mlp = nn.Linear(time_dim, out_c); self.res_conn = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t):
        h = self.norm1(self.relu(self.conv1(x))); h = h + self.time_mlp(self.relu(t)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1); h = self.norm2(self.relu(self.conv2(h))); return self.drop(h) + self.res_conn(x)

class CrossAttention3D(nn.Module):
    def __init__(self, chan, text_dim, heads=4):
        super().__init__(); self.heads = heads; self.scale = (chan // heads)**-0.5; self.to_q = nn.Conv3d(chan, chan, 1); self.to_k = nn.Linear(text_dim, chan); self.to_v = nn.Linear(text_dim, chan); self.to_out = nn.Conv3d(chan, chan, 1)
    def forward(self, x, ctx):
        b, c, d, h, w = x.shape; q = self.to_q(x).view(b, self.heads, c // self.heads, -1).permute(0, 1, 3, 2); k = self.to_k(ctx).view(b, self.heads, 1, c // self.heads); v = self.to_v(ctx).view(b, self.heads, 1, c // self.heads); scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale; attn = torch.softmax(scores, -1); out = torch.matmul(attn, v); out = out.permute(0, 1, 3, 2).reshape(b, c, d, h, w); return self.to_out(out) + x

class DownBlock(nn.Module):
    def __init__(self, i, o, t, x, d): super().__init__(); self.res = ResidualBlock3D(i, o, t, d); self.attn = CrossAttention3D(o, x); self.down = nn.Conv3d(o, o, 4, 2, 1)
    def forward(self, x, t, ctx): x = self.res(x, t); x = self.attn(x, ctx); return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, i, o, t, x, d): super().__init__(); self.up = nn.ConvTranspose3d(i, o, 4, 2, 1); self.res = ResidualBlock3D(o * 2, o, t, d); self.attn = CrossAttention3D(o, x)
    def forward(self, x, s, t, ctx): x = self.up(x); x = torch.cat([x, s], 1); x = self.res(x, t); return self.attn(x, ctx)

class UNet3D(nn.Module):
    def __init__(self, in_c, txt_d, time_d=128, base_c=64, drop=0.1):
        super().__init__(); self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_d), nn.Linear(time_d, time_d), nn.ReLU()); self.init_conv = nn.Conv3d(in_c, base_c, 3, 1, 1); self.down1 = DownBlock(base_c, base_c*2, time_d, txt_d, drop); self.down2 = DownBlock(base_c*2, base_c*4, time_d, txt_d, drop); self.bot1 = ResidualBlock3D(base_c*4, base_c*8, time_d, drop); self.bot_attn = CrossAttention3D(base_c*8, txt_d); self.bot2 = ResidualBlock3D(base_c*8, base_c*4, time_d, drop); self.up1 = UpBlock(base_c*4, base_c*2, time_d, txt_d, drop); self.up2 = UpBlock(base_c*2, base_c, time_d, txt_d, drop); self.final_conv = nn.Conv3d(base_c, in_c, 1)
    def forward(self, x, time, ctx): t = self.time_mlp(time); h1 = self.init_conv(x); h2 = self.down1(h1, t, ctx); h3 = self.down2(h2, t, ctx); h_bot = self.bot1(h3, t); h_bot = self.bot_attn(h_bot, ctx); h_bot = self.bot2(h_bot, t); h = self.up1(h_bot, h2, t, ctx); h = self.up2(h, h1, t, ctx); return self.final_conv(h)

class DiffusionScheduler:
    def __init__(self, steps=1000, start=0.0001, end=0.02, device=DEVICE):
        self.steps = steps; self.betas = torch.linspace(start, end, steps, device=device); self.alphas = 1.0 - self.betas; self.alpha_cumprod = torch.cumprod(self.alphas, 0)
    def add_noise(self, x, t):
        noise = torch.randn_like(x); sqrt_a = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1, 1); sqrt_1ma = (1-self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1, 1); return sqrt_a * x + sqrt_1ma * noise, noise

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    for param in clip_model.parameters():
        param.requires_grad = False
    print("CLIP model loaded.")

    os.makedirs("models", exist_ok=True)
    
    dataset = PreprocessedSchematicDataset(tensor_dir=PROCESSED_DIR, labels_path=LABELS_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    text_embed_dim = clip_model.text_projection.shape[-1]
    
    model = UNet3D(in_c=NUM_BLOCK_TYPES, txt_d=text_embed_dim).to(DEVICE)
    
    scheduler = DiffusionScheduler(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print(f"\n--- Starting Training ---")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, (clean_schematics, text_prompts) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            clean_schematics = clean_schematics.to(DEVICE)
            clean_schematics = clean_schematics * 2.0 - 1.0 
            
            text_embeddings = get_text_embedding(text_prompts, clip_model, device=DEVICE)
            
            t = torch.randint(0, scheduler.steps, (clean_schematics.shape[0],), device=DEVICE)
            noisy_schematics, noise = scheduler.add_noise(clean_schematics, t)
            predicted_noise = model(noisy_schematics, t, text_embeddings)
            
            loss = loss_fn(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            model_path = f"models/schematic_diffusion_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint to {model_path}")

    print("\n--- Training Complete ---")
    final_model_path = "models/schematic_diffusion_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
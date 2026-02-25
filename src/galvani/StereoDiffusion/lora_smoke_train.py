import os
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# -----------------------------
# Config (edit as needed)
# -----------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"

DATASET_ROOT = Path("/home/stud451/computergrafik_sing/dataset")  # <-- your dataset root
OUTPUT_DIR = Path("./lora_smoke_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 10           # smoke test: 1..10
BATCH_SIZE = 1           # keep 1 for smoke on cluster
LR = 1e-4
RANK = 8                 # LoRA rank
LORA_ALPHA = 8.0         # scaling
COND_SCALE = 0.10        # how strongly left_latent is injected
USE_FOCAL = True         # focal optional; if False -> only baseline used
SEED = 123

# SD constants
LATENT_SCALE = 0.18215   # SD v1 latent scaling
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dataset
# -----------------------------
class StereoFolderDataset(Dataset):
    """
    Expects folders that contain:
      left.jpg, right.jpg, meta.json
    meta.json contains baseline_m and optionally focal_mm
    """
    def __init__(self, root: Path):
        self.samples = []
        for meta_path in root.rglob("meta.json"):
            left = meta_path.parent / "left.jpg"
            right = meta_path.parent / "right.jpg"
            if left.exists() and right.exists():
                self.samples.append((left, right, meta_path))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {root}")

        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),                   # [0,1]
            transforms.Lambda(lambda x: x * 2 - 1),   # [-1,1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_p, right_p, meta_p = self.samples[idx]
        left_img = Image.open(left_p).convert("RGB")
        right_img = Image.open(right_p).convert("RGB")

        left = self.tf(left_img)
        right = self.tf(right_img)

        with open(meta_p, "r") as f:
            meta = json.load(f)

        baseline = float(meta.get("baseline_m", 0.0))
        focal = float(meta.get("focal_mm", 0.0))  # might be missing in some datasets

        # class name = top-level folder (e.g. Car/Dino/Piano)
        # .../dataset/Car/Camera_000/meta.json -> "Car"
        try:
            class_name = meta_p.relative_to(DATASET_ROOT).parts[0]
        except Exception:
            class_name = "unknown"

        return {
            "left": left,
            "right": right,
            "baseline": baseline,
            "focal": focal,
            "class_name": class_name,
            "path": str(meta_p.parent),
        }


# -----------------------------
# LoRA for Linear layers
# -----------------------------
class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank adapters.
    y = linear(x) + scale * (x @ A^T @ B^T)
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, device=DEVICE):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f = base.in_features
        out_f = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # LoRA params
        self.lora_A = nn.Linear(in_f, rank, bias=False, device=device)
        self.lora_B = nn.Linear(rank, out_f, bias=False, device=device)

        # init: A ~ N(0,0.01), B = 0 so start as no-op
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora_into_unet_attention(unet: UNet2DConditionModel, rank=8, alpha=8.0) -> List[str]:
    """
    Diffusers UNet has attention blocks with linear layers:
      to_q, to_k, to_v, to_out[0]
    We replace them with LoRALinear wrappers.
    Returns list of patched module names.
    """
    patched = []
    for name, module in unet.named_modules():
        # CrossAttention in diffusers has these attributes
        if hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear):
            module.to_q = LoRALinear(module.to_q, rank=rank, alpha=alpha, device=unet.device)
            patched.append(f"{name}.to_q")
        if hasattr(module, "to_k") and isinstance(module.to_k, nn.Linear):
            module.to_k = LoRALinear(module.to_k, rank=rank, alpha=alpha, device=unet.device)
            patched.append(f"{name}.to_k")
        if hasattr(module, "to_v") and isinstance(module.to_v, nn.Linear):
            module.to_v = LoRALinear(module.to_v, rank=rank, alpha=alpha, device=unet.device)
            patched.append(f"{name}.to_v")
        if hasattr(module, "to_out") and isinstance(module.to_out, nn.ModuleList) and len(module.to_out) > 0:
            if isinstance(module.to_out[0], nn.Linear):
                module.to_out[0] = LoRALinear(module.to_out[0], rank=rank, alpha=alpha, device=unet.device)
                patched.append(f"{name}.to_out.0")
    return patched


def lora_parameters(unet: nn.Module):
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


# -----------------------------
# Camera conditioning (baseline/focal)
# -----------------------------
class CameraCondMLP(nn.Module):
    """
    Maps (baseline, focal) -> 768-d vector that we add to text embeddings.
    focal can be disabled; then input_dim=1.
    """
    def __init__(self, use_focal: bool = True, out_dim: int = 768):
        super().__init__()
        self.use_focal = use_focal
        in_dim = 2 if use_focal else 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, baseline: torch.Tensor, focal: Optional[torch.Tensor] = None):
        if self.use_focal:
            assert focal is not None
            x = torch.stack([baseline, focal], dim=-1)
        else:
            x = baseline.unsqueeze(-1)
        return self.net(x)


# -----------------------------
# Utils
# -----------------------------
@torch.no_grad()
def encode_vae(vae: AutoencoderKL, img: torch.Tensor) -> torch.Tensor:
    """
    img: [B,3,H,W] in [-1,1]
    returns latents: [B,4,64,64]
    """
    lat = vae.encode(img).latent_dist.sample()
    return lat * LATENT_SCALE


def get_text_embeddings(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompts: List[str], device: str):
    text_in = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = text_encoder(text_in.input_ids.to(device))[0]
    return emb


def save_lora_checkpoint(unet: UNet2DConditionModel, cam_mlp: nn.Module, out_path: Path):
    """
    Saves only LoRA + camera MLP weights (small).
    """
    state = {"cam_mlp": cam_mlp.state_dict(), "lora": {}}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            state["lora"][name] = {
                "lora_A": module.lora_A.state_dict(),
                "lora_B": module.lora_B.state_dict(),
                "rank": module.rank,
                "alpha": module.alpha,
            }
    torch.save(state, out_path)


# -----------------------------
# Main training (smoke)
# -----------------------------
def main():
    torch.manual_seed(SEED)

    print(f"[Device] {DEVICE}")
    print(f"[Dataset] root={DATASET_ROOT}")
    ds = StereoFolderDataset(DATASET_ROOT)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # Load SD components
    print("[Load] tokenizer/text_encoder/vae/unet/scheduler ...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Freeze base modules
    text_encoder.eval(); vae.eval()
    for p in text_encoder.parameters(): p.requires_grad_(False)
    for p in vae.parameters(): p.requires_grad_(False)
    for p in unet.parameters(): p.requires_grad_(False)

    # Inject LoRA
    patched = inject_lora_into_unet_attention(unet, rank=RANK, alpha=LORA_ALPHA)
    print(f"[LoRA] patched {len(patched)} layers")
    if len(patched) == 0:
        raise RuntimeError("No attention layers patched. Diffusers version mismatch?")

    # Camera conditioning MLP (trainable)
    cam_mlp = CameraCondMLP(use_focal=USE_FOCAL, out_dim=768).to(DEVICE)

    # Optimizer over LoRA + cam_mlp
    params = list(lora_parameters(unet)) + list(cam_mlp.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    # Text embeddings: empty prompt (you can extend later)
    base_text = get_text_embeddings(tokenizer, text_encoder, [""] * BATCH_SIZE, DEVICE)  # [B,77,768]

    unet.train()
    cam_mlp.train()

    step = 0
    for batch in dl:
        left = batch["left"].to(DEVICE)    # [B,3,512,512] in [-1,1]
        right = batch["right"].to(DEVICE)

        baseline = torch.tensor(batch["baseline"], device=DEVICE, dtype=torch.float32)  # [B]
        focal = torch.tensor(batch["focal"], device=DEVICE, dtype=torch.float32)        # [B]

        # normalize conditioning (simple, works)
        # baseline choices up to ~0.14; focal up to ~40
        baseline_n = (baseline / 0.14).clamp(0, 2)
        focal_n = (focal / 40.0).clamp(0, 2)

        # VAE latents
        with torch.no_grad():
            left_lat = encode_vae(vae, left)
            right_lat = encode_vae(vae, right)

        # sample timestep
        t = torch.randint(0, scheduler.num_train_timesteps, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
        noise = torch.randn_like(right_lat)
        x_t = scheduler.add_noise(right_lat, noise, t)

        # left conditioning injection (simple but effective)
        x_in = x_t + COND_SCALE * left_lat

        # camera embedding added to text tokens
        cam_vec = cam_mlp(baseline_n, focal_n if USE_FOCAL else None)  # [B,768]
        text_emb = base_text + cam_vec.unsqueeze(1)                    # [B,77,768]

        # predict noise
        pred = unet(x_in, t, encoder_hidden_states=text_emb).sample
        loss = F.mse_loss(pred.float(), noise.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 1 == 0:
            print(f"[step {step:03d}] loss={loss.item():.6f}  baseline={baseline.tolist()} focal={focal.tolist()} path={batch['path'][0]}")

        step += 1
        if step >= MAX_STEPS:
            break

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "lora_smoke.pt"
    save_lora_checkpoint(unet, cam_mlp, ckpt_path)
    print(f"[Saved] {ckpt_path.resolve()}")
    print("DONE.")


if __name__ == "__main__":
    main()

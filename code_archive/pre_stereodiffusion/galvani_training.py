import os, json, math, random, torch
from collections import defaultdict
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import StereoScenesDataset
from sampler import BalancedByBaselineSampler


# directories
DATA_ROOT = "../../data/galvani"
DATA_DIR = f"{DATA_ROOT}/fixed_baselines"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
HF_CACHE = f"{DATA_ROOT}/hf_cache"
OUTPUT_DIR = f"{DATA_ROOT}/lora_out_2"

# settings
R_UNET     = 16
R_TE       = 8
LR         = 3e-5
BATCHSIZE  = 1
IMAGE_SIZE = (384, 384)

# training
STEPS_PER_EPOCH = 2000
MAX_STEPS = 25000
GRAD_ACCUM  = 4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda"


# pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  
pipe.enable_vae_tiling()
pipe.vae.enable_slicing()
pipe.enable_attention_slicing()
pipe.to("cuda")

for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
    m.to(device="cuda", dtype=torch.float16)


# LoRA
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# freeze base
for p in pipe.unet.parameters(): p.requires_grad_(False)
for p in pipe.vae.parameters():  p.requires_grad_(False)
for p in pipe.text_encoder.parameters(): p.requires_grad_(False)

# UNet-LoRA
lora_cfg_unet = LoraConfig(
    r=R_UNET,
    lora_alpha=R_UNET*2,
    lora_dropout=0.0,
    bias="none",
    target_modules=["to_q","to_k","to_v","to_out.0"],
    init_lora_weights="gaussian",
)
pipe.unet.add_adapter(lora_cfg_unet, adapter_name="stereo")
pipe.unet.set_adapters(["stereo"])
UNET_DTYPE = next(pipe.unet.parameters()).dtype
for n, p in pipe.unet.named_parameters():
    if "lora" in n: p.data = p.data.to(UNET_DTYPE)

# Text-Encoder-LoRA
te_lora = LoraConfig(
    r=R_TE, lora_alpha=R_TE*2, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","out_proj"]
)
pipe.text_encoder = get_peft_model(pipe.text_encoder, te_lora)
pipe.text_encoder.to(device="cuda", dtype=torch.float16)

# ---------------- token registration ----------------
ds = StereoScenesDataset(DATA_DIR, size=IMAGE_SIZE)

# registriere alle <B_xx>, plus <LEFT>, <RIGHT>
extra_tokens = sorted({s["btag"] for s in ds.samples} | {"<LEFT>", "<RIGHT>"})
added = pipe.tokenizer.add_tokens(list(extra_tokens))
if added > 0:
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

# ---------------- dataloader (A) ----------------
sampler = BalancedByBaselineSampler(ds.samples, batch_size=BATCHSIZE, steps_per_epoch=STEPS_PER_EPOCH)
dl = DataLoader(ds, batch_size=BATCHSIZE, sampler=sampler, collate_fn=lambda b: b, num_workers=0)

# ---------------- optimizer (B) ----------------
trainable = []
for n,p in pipe.unet.named_parameters():
    if "lora" in n and p.requires_grad: trainable.append(p)
for p in pipe.text_encoder.parameters():
    if p.requires_grad: trainable.append(p)

opt = optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))

# ---------------- encode helper ----------------
def get_prompt_embeds(pipe, prompts, device):
    try:
        enc = pipe.encode_prompt(
            prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        return enc[0] if isinstance(enc, tuple) else enc
    except Exception:
        tok = pipe.tokenizer(
            prompts, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        with torch.no_grad():
            enc = pipe.text_encoder(input_ids)[0]
        return enc

# ---------------- training (C): DDPM + SNR-Loss ----------------
train_sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
alphas_cumprod = train_sched.alphas_cumprod.to("cuda")  # [T]

vae, unet = pipe.vae, pipe.unet
vae.eval()
pipe.text_encoder.eval()
unet.train()

step = 0
pbar = tqdm(total=MAX_STEPS, desc="train")
opt.zero_grad(set_to_none=True)

while step < MAX_STEPS:
    for batch in dl:
        images  = [s["image"]  for s in batch]
        prompts = [s["prompt"] for s in batch]

        # leichte, sichere Augmentation (optional)
        # -> hier weggelassen, falls gewünscht: kleine Helligkeits-/Kontrastjitter vor preprocess

        with torch.no_grad():
            vae_dtype = next(vae.parameters()).dtype
            px = pipe.image_processor.preprocess(images).to("cuda", dtype=vae_dtype)
            latents = vae.encode(px).latent_dist.sample().to(UNET_DTYPE) * 0.18215

            noise = torch.randn_like(latents, dtype=UNET_DTYPE, device="cuda")
            t = torch.randint(0, train_sched.config.num_train_timesteps, (latents.size(0),), device="cuda").long()
            noisy_latents = train_sched.add_noise(latents, noise, t)

            prompt_embeds = get_prompt_embeds(pipe, prompts, "cuda").to("cuda", dtype=UNET_DTYPE)

        pred = unet(noisy_latents, t, encoder_hidden_states=prompt_embeds).sample

        # SNR-Loss
        snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
        gamma = 5.0
        loss_weight = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1.0)
        per_ex = (pred - noise).pow(2).mean(dim=(1,2,3))
        loss = (loss_weight * per_ex).mean()
        print(f"pred = unet({noisy_latents}, t, encoder_hidden_states={prompt_embeds}).sample = {pred}")
        print(f"\tstep: {step}/{MAX_STEPS}\tloss: {loss} (mean({loss_weight} * {per_ex}))")

        if torch.isnan(loss):
            exit()

        (loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
        if step >= MAX_STEPS:
            break

pbar.close()
print("Training done.")


# Tokenizer (neue Tokens)
pipe.tokenizer.save_pretrained(OUTPUT_DIR)

# UNet-LoRA
pipe.unet.set_adapters(["stereo"])
pipe.save_lora_weights(
    OUTPUT_DIR,
    unet_lora_layers=pipe.unet,
    weight_name="stereo_unet.safetensors",
    safe_serialization=True,
)

# Text-Encoder-LoRA
from peft import get_peft_model_state_dict
te_lora_sd = get_peft_model_state_dict(pipe.text_encoder)
torch.save(te_lora_sd, os.path.join(OUTPUT_DIR, "stereo_te.safetensors"))

print("Saved UNet-LoRA ->", os.path.join(OUTPUT_DIR, "stereo_unet.safetensors"))
print("Saved TE-LoRA   ->", os.path.join(OUTPUT_DIR, "stereo_te.safetensors"))
print("Tokenizer saved ->", OUTPUT_DIR)
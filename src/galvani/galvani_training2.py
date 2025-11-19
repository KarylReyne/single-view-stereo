import os, json, math, random, torch
from collections import defaultdict
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tueplots.constants.color import rgb

from dataset import StereoScenesDataset
from sampler import BalancedByBaselineSampler
from plotting_util import get_next_tue_plot_color


# directories
DATA_ROOT = "../../data/galvani"
DATA_DIR = f"{DATA_ROOT}/fixed_baselines"
# MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
HF_CACHE = f"{DATA_ROOT}/hf_cache"
OUTPUT_DIR = f"{DATA_ROOT}/lora_out2"

# settings
R_UNET     = 16
UNET_DROPOUT = 0.01
R_TE       = 8
TEXTENCODER_DROPOUT = 0.01
LR         = 3e-5
BATCHSIZE  = 16 # 1
IMAGE_SIZE = (384, 384)

# training
STEPS_PER_EPOCH = 2000
# MAX_STEPS = 25000
MAX_STEPS = 1000
GRAD_ACCUM  = 4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda"
DTYPE = torch.float32


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
pipe.to(DEVICE)

for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
    m.to(device=DEVICE, dtype=DTYPE)


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
    lora_dropout=UNET_DROPOUT,
    bias="none",
    target_modules=["to_q","to_k","to_v","to_out.0"],
    init_lora_weights="gaussian",
)
pipe.unet.add_adapter(lora_cfg_unet, adapter_name="stereo")
pipe.unet.set_adapters(["stereo"])

# UNET_DTYPE = next(pipe.unet.parameters()).dtype
for n, p in pipe.unet.named_parameters():
    if "lora" in n: 
        p.data = p.data.to(DTYPE)

# Text-Encoder-LoRA
lora_cfg_text_encoder = LoraConfig(
    r=R_TE, 
    lora_alpha=R_TE*2, 
    lora_dropout=TEXTENCODER_DROPOUT, 
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","out_proj"]
)
pipe.text_encoder = get_peft_model(pipe.text_encoder, lora_cfg_text_encoder)
pipe.text_encoder.to(device=DEVICE, dtype=DTYPE)

# ---------------- token registration ----------------
ds = StereoScenesDataset(DATA_DIR, size=IMAGE_SIZE)

# registriere alle <B_xx>, plus <LEFT>, <RIGHT>
extra_tokens = sorted({s["btag"] for s in ds.samples} | {"<LEFT>", "<RIGHT>"})
added = pipe.tokenizer.add_tokens(list(extra_tokens))
if added > 0:
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

# ---------------- dataloader (A) ----------------
sampler = BalancedByBaselineSampler(
    ds.samples, 
    batch_size=BATCHSIZE, 
    steps_per_epoch=STEPS_PER_EPOCH
)
dl = DataLoader(
    ds, 
    batch_size=BATCHSIZE, 
    sampler=sampler, 
    collate_fn=lambda b: b, # why?!
    num_workers=0
)

# ---------------- optimizer (B) ----------------
trainable = []

for n,p in pipe.unet.named_parameters():
    if "lora" in n and p.requires_grad: 
        trainable.append(p)

for p in pipe.text_encoder.parameters():
    if p.requires_grad: 
        trainable.append(p)

opt = optim.AdamW(
    trainable, 
    lr=LR, 
    weight_decay=WEIGHT_DECAY, 
    betas=(0.9, 0.99)
)

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
    # tokens = pipe.tokenizer(
    #     prompts, 
    #     padding="max_length",
    #     max_length=pipe.tokenizer.model_max_length,
    #     truncation=True, 
    #     return_tensors="pt",
    # )
    # input_ids = tokens.input_ids.to(device)
    # with torch.no_grad():
    #     enc = pipe.text_encoder(input_ids)[0]
    # return enc

# ---------------- training (C): DDPM + SNR-Loss ----------------
train_sched = DDPMScheduler(
    num_train_timesteps=1000, 
    beta_schedule="scaled_linear"
)
alphas_cumprod = train_sched.alphas_cumprod.to(DEVICE)

vae, unet = pipe.vae, pipe.unet
vae.eval()
pipe.text_encoder.eval()
unet.train()

step = 0
pbar = tqdm(total=MAX_STEPS, desc="train")
opt.zero_grad(set_to_none=True)

loss_per_step = []
steps = []

while step < MAX_STEPS:
    for batch in dl:
        images  = [s["image"]  for s in batch]
        prompts = [s["prompt"] for s in batch]

        with torch.no_grad():
            vae_dtype = next(vae.parameters()).dtype
            px = pipe.image_processor.preprocess(images).to(DEVICE, dtype=vae_dtype)
            latents = vae.encode(px).latent_dist.sample().to(DTYPE) * 0.18215

            noise = torch.randn_like(latents, dtype=DTYPE, device=DEVICE)
            t = torch.randint(0, train_sched.config.num_train_timesteps, (latents.size(0),), device=DEVICE).long()
            noisy_latents = train_sched.add_noise(latents, noise, t)

            prompt_embeds = get_prompt_embeds(pipe, prompts, DEVICE).to(DEVICE, dtype=DTYPE)
            
        pred = unet(noisy_latents, t, encoder_hidden_states=prompt_embeds).sample

        # print(pred)

        # SNR-Loss
        snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
        gamma = 5.0
        loss_weight = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1.0)
        per_ex = (pred - noise).pow(2).mean(dim=(1,2,3))
        loss = (loss_weight * per_ex).mean()
        
        # print(f"\tstep: {step}/{MAX_STEPS}\tloss: {loss} (mean({loss_weight} * {per_ex}))")

        # recording loss for plotting
        if step % 4 == 0:
            loss_per_step.append(float(loss.cpu().detach().numpy()))
            steps.append(step)

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

# loss plotting
MS = 2
LW = 1.3
FONTSIZE = 10
# AXES_ASPECT = 10
GRID_LW = 0.5

fig, ax = plt.subplots()
line, = ax.plot(
    steps,
    loss_per_step,
    "-",
    ms=MS,
    lw=LW,
    color=get_next_tue_plot_color(0),
    label="SNR loss"
)
legend = ax.legend(
    bbox_to_anchor=(1.01, 1.01), 
    fontsize=FONTSIZE,
    # title="loss during training"
)
legend.get_frame().set_edgecolor(color=rgb.tue_gray)

ax.set_title("loss during training")
ax.set_xlabel("training step", fontsize=FONTSIZE)
ax.set_ylabel("loss", fontsize=FONTSIZE)
# ax.set_aspect(AXES_ASPECT)
ax.grid(axis="both", color=rgb.tue_gray, linewidth=GRID_LW)

fig.tight_layout()

fig.savefig(OUTPUT_DIR+f"/loss_during_training.pdf")


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
import os, torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


# configuration
MODEL_ID  = "runwayml/stable-diffusion-v1-5"
DATA_ROOT = "../../data/galvani"
CKPT_DIR  = f"{DATA_ROOT}/lora_out"
W, H      = 512, 512
DEVICE    = "cuda"


# pipeline
txt2img = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
).to(DEVICE)
img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
).to(DEVICE)
for p in (txt2img, img2img):
    p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
    p.enable_vae_tiling(); p.vae.enable_slicing(); p.enable_attention_slicing()

# tokenizer
tok = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=False)
for p in (txt2img, img2img):
    p.tokenizer = tok
    p.text_encoder.resize_token_embeddings(len(tok))

# Load LoRa
def load_unet_any(pipeline, ckpt_dir):
    lora_path = os.path.join(ckpt_dir, "stereo_unet.safetensors")
    full_paths = [os.path.join(ckpt_dir, "stereo.safetensors"), lora_path]

    if os.path.exists(lora_path):
        try:
            pipeline.load_lora_weights(ckpt_dir, weight_name="stereo_unet.safetensors", adapter_name="stereo")
            pipeline.set_adapters(["stereo"])
            pipeline.set_lora_scale(1.0)
            print("[INFO]", "Loaded UNet LoRA via load_lora_weights:", lora_path)
            return "lora"
        except Exception as e:
            print("[WARN]", "LoRA load failed, try FULL UNet fallback.", str(e))

    # fallback
    for fpath in full_paths:
        if not os.path.exists(fpath): continue
        sd = load_file(fpath)
        keys = list(sd.keys())
        if any(k.startswith("unet.") for k in keys):
            sd_unet = {k[len("unet."):]: v for k, v in sd.items() if k.startswith("unet.")}
        else:
            sd_unet = sd
        missing, unexpected = pipeline.unet.load_state_dict(sd_unet, strict=False)
        if missing or unexpected:
            print("[WARN]", "UNET load_state_dict missing:", list(missing)[:5], "unexpected:", list(unexpected)[:5])
        pipeline.unet.to(DEVICE, dtype=torch.float16)
        print("[INFO]", "Loaded FULL UNet from:", fpath)
        return "full"

    raise FileNotFoundError("Kein UNet-Checkpoint gefunden (stereo_unet.safetensors oder stereo.safetensors).")

mode_txt = load_unet_any(txt2img, CKPT_DIR)
mode_img = load_unet_any(img2img, CKPT_DIR)

# text encoder lora
te_lora_path = os.path.join(CKPT_DIR, "stereo_te.safetensors")
if os.path.exists(te_lora_path):
    te_cfg = LoraConfig(
      r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
      target_modules=["q_proj","k_proj","v_proj","out_proj"]
    )

    for p in (txt2img, img2img):
        p.text_encoder = get_peft_model(p.text_encoder, te_cfg).to(DEVICE, dtype=torch.float16)
        te_state = torch.load(te_lora_path, map_location=DEVICE)
        set_peft_model_state_dict(p.text_encoder, te_state, adapter_name="default")
    print("[INFO]", "Loaded TE LoRA:", te_lora_path)
else:
    print("[WARN]", "Kein stereo_te.safetensors – Steuer-Tokens wirken ggf. schwächer.")

# generation
def generate_stereo_img2img(
    b_cm=8, fov=60, core="street scene",
    steps=28, guidance=5.0, strength=0.25, seed=123, out=f"{DATA_ROOT}/output"
):
    os.makedirs(out, exist_ok=True)
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    btag = f"<B_{int(b_cm):02d}>"
    promptL = f"{btag} <LEFT> {core}"
    promptR = f"{btag} <RIGHT> {core}"

    left = txt2img(
        prompt=promptL,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=H, width=W,
        generator=g
    ).images[0]

    right = img2img(
        prompt=promptR,
        image=left,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=g
    ).images[0]

    left_path  = f"{out}/stereo_B{b_cm}_left.png"
    right_path = f"{out}/stereo_B{b_cm}_right.png"
    side_path  = f"{out}/stereo_B{b_cm}_side.png"

    left.save(left_path); right.save(right_path)
    side = Image.new("RGB", (W * 2, H)); side.paste(left, (0, 0)); side.paste(right, (W, 0))
    side.save(side_path)

    print(f"[OK] Saved:\n  {left_path}\n  {right_path}\n  {side_path}")
    return left, right, side


# example
_ = generate_stereo_img2img(
    b_cm=20, core="Horse walking on a street", steps=28, guidance=5.0, strength=0.25, seed=42
)
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import sys
from typing import Optional, List
from skimage.transform import resize
from diffusers import StableDiffusionPipeline, DDIMScheduler
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn.functional import normalize
import random

sys.path.append('./DensePredictionTransformer')
from DensePredictionTransformer.dpt.models import DPTDepthModel
from stereoutils import stereo_shift_torch, norm_depth, BNAttention, register_attention_editor_diffusers, load_512, load_exr

sys.path.append('./PromptToPrompt')
import ptp_utils
from ptp_null_text import AttentionStore, make_controller

sys.path.append('..')
from QwenPromptInterpreter.prompt2float import interpret_prompt
from misc_util import get_config, add_subfolder_to_save_prefix, save_config
from ptp_save_util import save_images, save_cross_attention, save_hist_from_array, save_generated_stereoimages
from stereodiffusion_nti import EmptyControl, NullInversion
from lora_smoke_train import inject_lora_into_unet_attention, CameraCondMLP, LoRALinear, lora_parameters


def run_and_display(
    ldm_stable, prompts, controller, disparity, inf_config,
    cam_mlp, camera_params, # NEU hinzufügen
    latents=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True,
):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        prev_config = inf_config["output_prefix"]
        inf_config["output_prefix"] = os.sep.join([inf_config["output_prefix"], "without-ptp"])
        images, latent = run_and_display(
            ldm_stable, 
            ["", ""],
            EmptyControl(),
            disparity,
            inf_config,
            cam_mlp,
            camera_params,
            latents=[torch.concat([latent,latent],0)],
            run_baseline=False, 
            generator=generator, 
            uncond_embeddings=uncond_embeddings,
            verbose=verbose
        )
        inf_config["output_prefix"] = prev_config
        print("with prompt-to-prompt")
    images, latent = text2stereoimage_ldm_stable(
        ldm_stable, 
        prompts,
        controller,
        disparity,
        inf_config,
        cam_mlp=cam_mlp,                    # NEU
        camera_params=camera_params,
        uncond_embeddings=uncond_embeddings,
        latents=latents,
        verbose=verbose
    )
    save_generated_stereoimages(images, inf_config) # unaffected by verbose, unlike every other save
    return images, latent


def _get_left_latent(curr_latents, latents, i, t, model, inf_config, verbose=False):
    left_latent = None
    if inf_config["align_right_with_gt"]: # noised gt left from ddim inversion
        left_latent = latents[-i] 
        if verbose:
            _left_latent = ptp_utils.latent2image(model.vae, left_latent)
            save_images(_left_latent, f'{inf_config["output_prefix"]}left-latent_from_gt_at-t={t}-i={i}.png')
    else:
        left_latent = curr_latents[:1] # generated left
        if verbose:
            _left_latent = ptp_utils.latent2image(model.vae, left_latent)
            save_images(_left_latent, f'{inf_config["output_prefix"]}left-latent_from_gen_at-t={t}-i={i}.png')
    return left_latent


@torch.no_grad()
def text2stereoimage_ldm_stable(
    model,
    prompts: List[str],
    controller,
    disparity,
    inf_config,
    cam_mlp,          
    camera_params,       
    generator: Optional[torch.Generator] = None,
    latents: Optional[List[torch.FloatTensor]] = None,
    uncond_embeddings=None,
    latents_editing_freq=10,
    return_type='image',
    verbose=False
):
    if controller.__class__.__name__ == "BNAttention":
        register_attention_editor_diffusers(model, controller) # StereoDiffusion
    else:
        ptp_utils.register_attention_control(model, controller) # Prompt-to-Prompt

    batch_size = len(prompts)
    height = width = 512
    
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    if inf_config["use_lora_weights_for_shifting"]:
        b_n = torch.tensor([camera_params[0] / 0.14], device=model.device).float()
        f_n = torch.tensor([camera_params[1] / 40.0], device=model.device).float()

        with torch.no_grad():
            # cam_mlp muss als Parameter an diese Funktion übergeben werden!
            cam_vec = cam_mlp(b_n, f_n) # Ergebnis: [1, 768]
            
            # In StereoDiffusion ist batch_size=2: [Links, Rechts]
            # Wir addieren das gelernten Verschiebe-Signal NUR auf das rechte Bild (Index 1)
            if text_embeddings.shape[0] > 1:
                text_embeddings[1:2] = text_embeddings[1:2] + cam_vec.unsqueeze(1)
    
    latent, curr_latents = ptp_utils.init_latent(latents[-1], model, height, width, generator, batch_size)

    if verbose:
        _latents_init = ptp_utils.latent2image(model.vae, curr_latents)
        save_images(_latents_init, f'{inf_config["output_prefix"]}initial_latents.png')

    model.scheduler.set_timesteps(inf_config["num_ddim_steps"])

    for i, t in enumerate(tqdm(model.scheduler.timesteps[-inf_config["num_ddim_steps"]:])):

        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        curr_latents = ptp_utils.diffusion_step(model, controller, curr_latents, context, t, inf_config["guidance_scale"], low_resource=inf_config["low_resource"])
        
        if (i % latents_editing_freq == 0) and verbose:
            _latents_at_t = ptp_utils.latent2image(model.vae, curr_latents)
            save_images(_latents_at_t, f'{inf_config["output_prefix"]}latents_at_t={t}.png')
        
        # also reconstruct a right-side stereo image (StereoDiffusion)
        if i == latents_editing_freq:

            if not inf_config["use_lora_weights_for_shifting"]:
                if isinstance(disparity,torch.Tensor):
                    disparity = torch.nn.functional.interpolate(disparity.unsqueeze(1),size=[64,64],mode="bicubic",align_corners=False,).squeeze(1)
                elif isinstance(disparity,np.ndarray):
                    disparity = resize(disparity,(64,64))

            left_latent = _get_left_latent(curr_latents, latents, i, t, model, inf_config, verbose=verbose)
            latents_current = stereo_shift_torch(
                left_latent, 
                disparity, 
                scale_factor_percent=8
            )
            latents_current = latents_current[1:] # latents_current <- right latent
            curr_latents = torch.cat([curr_latents[:1], latents_current], 0) # [left latent, left latent shifted (right latent)]

            if verbose:
                _latents_at_t = ptp_utils.latent2image(model.vae, curr_latents)
                save_images(_latents_at_t, f'{inf_config["output_prefix"]}latents-after-shift_at_t={t}.png')

            mask = latents_current[:,0,...] != 0
            mask = rearrange(mask,'b h w ->b () h w').repeat(1,4,1,1)
            noise = torch.randn_like(curr_latents)

            if verbose:
                _mask = ptp_utils.latent2image(model.vae, mask)
                save_images(_mask, f'{inf_config["output_prefix"]}denoising-mask.png')

            if inf_config["stereodiffusion_deblur"]:
                curr_latents[1:][~mask] = noise[1:][~mask]
                curr_latents[1:][mask] = latents_current[mask]

        if  (i > latents_editing_freq and i % latents_editing_freq == 0) and not inf_config["use_lora_weights_for_shifting"]:

            left_latent = _get_left_latent(curr_latents, latents, i, t, model, inf_config, verbose=verbose)
            latents_current = stereo_shift_torch(
                left_latent,
                disparity, 
                scale_factor_percent=8
            )
            latents_current = latents_current[1:] # latents_current <- right latent
            curr_latents[1:][mask] = latents_current[mask] # prev right latent * mask <- curr right latent * mask
            
            if verbose:
                _latents_masked = ptp_utils.latent2image(model.vae, curr_latents)
                save_images(_latents_masked, f'{inf_config["output_prefix"]}latents-with-applied-mask_at_t={t}.png')

    if inf_config["align_right_with_gt"]:
        left_latent = latents[0]
        curr_latents[:1] = left_latent
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, curr_latents)
    else:
        image = latents
    return image, latent


def load_my_trained_lora(unet, cam_mlp, path):
    print(f"Lade trainierte Gewichte von {path}...")
    checkpoint = torch.load(path, map_location=unet.device)
    
    # 1. MLP Gewichte laden
    cam_mlp.load_state_dict(checkpoint["cam_mlp"])
    
    # 2. LoRA Gewichte in die Layer kopieren
    lora_state = checkpoint["lora"]
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            module.lora_A.load_state_dict(lora_state[name]["lora_A"])
            module.lora_B.load_state_dict(lora_state[name]["lora_B"])
    print("Gewichte erfolgreich geladen.")


def get_baseline_and_focal_length(img_path, inf_config, qpi_config, baseline_prompt=None, metadata=None, verbose=False):
    # custom baseline distance and focal length
    prompted_baseline = None
    focal_length = None

    # set baseline via prompt
    if inf_config["depthmap_from_prompt"]:
        assert baseline_prompt != None
        prompted_baseline, focal_length = interpret_prompt(baseline_prompt, qpi_config)
        if prompted_baseline == 0.0:
            if verbose:
                print(f"[DEPTHMAP_FROM_PROMPT] baseline can`t be {prompted_baseline}! setting B=1e-8")
            prompted_baseline = 1e-8

    # testing depthmap generation from sensor data (blender)
    if inf_config["depthmap_from_sensor"]:
        # import metadata file
        if metadata is None:
            try:
                metadata_path = img_path.split("/")[:-1]
                metadata_path.append("meta.json")
                metadata_path = "/".join(metadata_path)
                metadata = get_config(path=metadata_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"meta data file for '{img_path}' not found at '{metadata_path}'.")
        prompted_baseline = metadata["baseline_m"]
        focal_length = metadata["focal_mm"]

    return prompted_baseline, focal_length


def estimate_disparity_from_gt(
        image_gt,
        camera_params,
        depthmodel,
        disparitymodel,
        inf_config,
        verbose=False
    ):
    maps_folder = add_subfolder_to_save_prefix(inf_config, "disparity_maps")

    image_gt_ = torch.tensor(np.expand_dims(image_gt/255,0).transpose(0,3,1,2)/255, device=inf_config["device"], dtype=torch.float32)
    with torch.no_grad():
        depth_prediction = depthmodel.forward(image_gt_)
        disparity_prediction = disparitymodel.forward(image_gt_)
    
    # estimate disparity/depth
    prompted_baseline, focal_length = camera_params
    depth = norm_depth(depth_prediction)
    depth_to_disparity = norm_depth((focal_length*prompted_baseline)/depth_prediction)
    disparity = norm_depth(disparity_prediction)

    # print estimated disparity/depth
    maps = [depth, depth_to_disparity, disparity]
    for i in range(len(maps)):
        map = rearrange(maps[i], 'c h w -> (c h) w')
        map = map.cpu().numpy()
        map = np.uint8(map*255)
        maps[i] = map
    Image.fromarray(maps[0]).save(f'{maps_folder}DPT-depth.png')
    Image.fromarray(maps[1]).save(f'{maps_folder}DPT-depth-to-disparity.png')
    Image.fromarray(maps[2]).save(f'{maps_folder}DPT-disparity.png')
    if verbose:
        save_hist_from_array(maps[0], f'{maps_folder}DPT-depth_hist.png', title=r"Histogram of $Z_{DPT}$", color_idx=1)
        save_hist_from_array(maps[1], f'{maps_folder}DPT-depth-to-disparity_hist.png', title=r"Histogram of $D(B_{sensor},f_{sensor},Z_{DPT})$", color_idx=3)
        save_hist_from_array(maps[2], f'{maps_folder}DPT-disparity_hist.png', title=r"Histogram of $D_{DPT}$", color_idx=0)

    del depthmodel
    del disparitymodel

    return depth_to_disparity, disparity


def run_inv_sd(
    img_path,
    prompts,
    camera_params,
    ldm_stable,
    depthmodel,
    disparitymodel,
    inf_config,
    cam_mlp=None,
    verbose=False
):
    if inf_config["use_lora_weights_for_shifting"]:
        assert cam_mlp != None

    reconstruction_prompt = prompts[0]
    image = load_512(img_path)
    null_inversion = NullInversion(ldm_stable, inf_config)
    (image_gt, image_enc), ddim_latents, uncond_embeddings = null_inversion.invert(
        image, 
        reconstruction_prompt, 
        offsets=(0,0,200,0), 
        verbose=verbose
    )
    del null_inversion

    depth_to_disparity, disparity = estimate_disparity_from_gt(
        image_gt,
        camera_params,
        depthmodel,
        disparitymodel,
        inf_config,
        verbose=verbose
    )

    # select disparity map
    if inf_config["use_depth-to-disparity"]:
        disp = depth_to_disparity
    else:
        disp = disparity

    # select attention controller
    if inf_config["use_cross_attn"]:
        controller = AttentionStore(low_resource=inf_config["low_resource"])
    else:
        controller = BNAttention(
            start_step=inf_config["stereodiffusion_attn_steps_start"], 
            total_steps=inf_config["stereodiffusion_attn_steps_total"], 
            direction=inf_config["stereodiffusion_attn_direction"]
        )

    image_inv, latent = run_and_display(
        ldm_stable,
        prompts, 
        controller,
        disp, 
        inf_config,
        cam_mlp=cam_mlp,                  # Korrekte Übergabe
        camera_params=camera_params,
        run_baseline=False,
        latents=ddim_latents, 
        uncond_embeddings=uncond_embeddings,
        verbose=verbose
    )
    if verbose:
        print("saving attention score maps...", end="")
        save_cross_attention([prompts[1]], ldm_stable.tokenizer, controller, 16, ["up", "down"], f'{inf_config["output_prefix"]}attention.png')
        print("done")

    return image_inv, latent, depth_to_disparity, disparity


def get_models(inf_config):
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        clip_sample=False, 
        set_alpha_to_one=False,
        steps_offset=1
    )

    ldm_stable = StableDiffusionPipeline.from_pretrained(
        inf_config["stablediffusion_model"], 
        scheduler=scheduler
    ).to(inf_config["device"])
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    
    cam_mlp = None
    if inf_config["use_lora_weights_for_shifting"]:
        inject_lora_into_unet_attention(ldm_stable.unet, rank=8, alpha=8.0)
        cam_mlp = CameraCondMLP(use_focal=True).to(inf_config["device"])
        lora_path = inf_config["lora_path"]
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at {lora_path}")
        load_my_trained_lora(ldm_stable.unet, cam_mlp, lora_path)

    depthmodel = DPTDepthModel(
        path=inf_config["depthmodel_path"],
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        invert=True
    ).to(inf_config["device"])

    disparitymodel = DPTDepthModel(
        path=inf_config["depthmodel_path"],
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        invert=False
    ).to(inf_config["device"])

    return ldm_stable, depthmodel, disparitymodel, cam_mlp 

def get_dataset_samples_from_folder_tree(root_ptr, depth=1, files_to_get=["left.jpg", "right.jpg", "meta.json"], shuffle=False, shuffle_seed=42, max_samples=None, verbose=True):
    def get_subfolder_paths(root_folders):
        folders = []
        for root_folder in root_folders:
            for _, dirs, _ in os.walk(root_folder, followlinks=False):
                if len(dirs) == 0: break
                for dir in dirs:
                    folders.append(os.sep.join([root_folder, dir]))
                if len(folders) >= len(dirs): break
        return folders
    
    if verbose: print(f"loading dataset at '{root_ptr}'...")

    current_depth = depth
    samples_paths = get_subfolder_paths([root_ptr])
    while current_depth > 1:
        samples_paths = get_subfolder_paths(samples_paths)
        current_depth -= 1

    if verbose: print()
    counter = 1
    skipped = 0
    samples = []
    for sample_path in samples_paths:
        if verbose:
            sys.stdout.write("\033[F")
            print(f"processing sample {counter}/{len(samples_paths)} ({skipped} skipped)")

        # make sure all requested files exist
        files = None
        for _, _, _files in os.walk(sample_path):
            files = {f: None for f in _files} # for O(1) search
        try:
            for f in files_to_get:
                _ = files[f]
        except KeyError:
            skipped += 1
            continue
        
        # samples[i] <- {
        #   file_to_get: path_to_file
        #   "sample_path": path/to/sample/in/dataset/
        # }
        sample_dict = {f: os.sep.join([sample_path, f]) for f in files_to_get}
        sample_dict["sample_path"] = sample_path.replace(root_ptr, "")
        samples.append(sample_dict)
        counter += 1

    samples_indices = np.arange(len(samples))
    if shuffle:
        random.Random(shuffle_seed).shuffle(samples_indices)

    if max_samples != None:
        samples_indices = samples_indices[:max_samples]
    
    return samples, samples_indices


if __name__ == "__main__":
    # inf_config = get_config(path="../cfg/inference_config.json")

    # --- done --- seed=42, 1k samples
    # eval1 no prompt | disparity | uni-directional | untrained | gen aligned
    # inf_config = get_config(path="../cfg/eval1_config.json")

    # eval3 prompt | disparity | cross | untrained | gen aligned
    # inf_config = get_config(path="../cfg/eval3_config.json")

    # eval4 prompt | disparity | uni-directional | untrained | gen aligned
    # inf_config = get_config(path="../cfg/eval4_config.json")

    # eval2 no prompt | depth-to-disparity | uni-directional | untrained | gen aligned
    # inf_config = get_config(path="../cfg/eval2_config.json")

    # eval6 no prompt | disparity | uni-directional | trained | gen aligned
    # inf_config = get_config(path="../cfg/eval6_config.json")

    # eval7 no prompt | disparity | uni-directional | untrained | gt aligned 
    # inf_config = get_config(path="../cfg/eval7_config.json")


    # --- in progress --- seed=42, 1k samples
    # at eval1
    # eval8 prompt | disparity | uni-directional | untrained | gt aligned
    inf_config = get_config(path="../cfg/eval8_config.json")

    # at eval2
    # eval5 prompt | disparity | bi-directional | untrained | gen aligned
    # inf_config = get_config(path="../cfg/eval5_config.json")

    # at eval3

    # at eval4


    # --- todo --- seed=42, 1k samples
    

    qpi_config = get_config(path="../cfg/qwen_config.json")
    os.makedirs(inf_config["output_prefix"], exist_ok=True)

    # verbose = inf_config["verbose"]

    # --- get attention maps for the report ---
    inf_config["dataset_path"] = "../../../data/galvani/one_sample"
    
    verbose = True
    # --- end ----

    ldm_stable, depthmodel, disparitymodel, cam_mlp = get_models(inf_config)

    samples, samples_indices = get_dataset_samples_from_folder_tree(
        inf_config["dataset_path"],
        depth=inf_config["dataset_depth"],
        files_to_get=["left.jpg", "right.jpg", "meta.json", "disparity.exr"],
        shuffle=inf_config["shuffle_dataset"],
        shuffle_seed=inf_config["shuffle_dataset_seed"],
        max_samples=inf_config["dataset_max_samples"],
        verbose=True
    )
    root_output_prefix = inf_config["output_prefix"]

    psnr = lambda x, y: float(peak_signal_noise_ratio(x, y, data_range=255))
    ssim = lambda x, y: float(structural_similarity(x, y, data_range=255, channel_axis=-1))
    _lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(inf_config["device"])
    
    def lpips(x, y, _lpips=_lpips, device=inf_config["device"]):
        if isinstance(x, np.ndarray): x = torch.Tensor(x).to(device)
        if isinstance(y, np.ndarray): y = torch.Tensor(y).to(device)
        x = normalize(x)
        y = normalize(y)
        if x.dim() == 2 and y.dim() == 3: # disparity maps
            x = rearrange(x, "h w -> () h w").repeat(1,3,1,1)
            y = y.repeat(1,3,1,1) # 1 h w -> 1 3 h w
        else: # left/right images
            x = rearrange(x, "h w c -> () c h w")
            y = rearrange(y, "h w c -> () c h w")
        return float(_lpips(x, y))
    
    all_left_psnr_scores = []
    all_left_ssim_scores = []
    all_left_lpips_scores = []
    all_right_psnr_scores = []
    all_right_ssim_scores = []
    all_right_lpips_scores = []
    all_disp_lpips_scores = []
    all_depth_to_disp_lpips_scores = []
    counter = 1

    for i in samples_indices:
        sample = samples[i]

        print(f"--- processing sample {counter}/{len(samples_indices)} ---")

        left_img_path = sample["left.jpg"]
        metadata = get_config(sample["meta.json"])

        # override the output_prefix to replicate the dataset's folder tree
        inf_config["output_prefix"] += sample["sample_path"]+os.sep
        os.makedirs(inf_config["output_prefix"], exist_ok=True)

        if inf_config["use_baseline_prompt"]:
            baseline_prompt = inf_config["baseline_prompt"]
        else:
            baseline_prompt = None # => get params from metadata instead

        prompted_baseline, focal_length = get_baseline_and_focal_length(
            left_img_path,  
            inf_config,
            qpi_config,
            baseline_prompt=baseline_prompt,
            metadata=metadata,
            verbose=verbose
        )

        if inf_config["use_conditioning_prompt"]:
            reconstruction_prompt = f"{metadata['desc'].rstrip('.')}, captured with a stereo camera with baseline distance 0.0 and focal length {focal_length:.2f}"
            conditioning_prompt = f"{metadata['desc'].rstrip('.')}, captured with a stereo camera with baseline distance {prompted_baseline:.2f} and focal length {focal_length:.2f}"
        else:
            reconstruction_prompt = f""
            conditioning_prompt = f""

        prompts = [
            reconstruction_prompt,
            conditioning_prompt
        ]
        camera_params = [
            prompted_baseline,
            focal_length
        ]

        sample_config = {
            "sample_index": int(i),
            "sample_path": sample["sample_path"],
            "baseline_prompt": baseline_prompt,
            "reconstruction_prompt": reconstruction_prompt,
            "conditioning_prompt": conditioning_prompt,
            "prompted_baseline": prompted_baseline,
            "focal_length": focal_length
        }

        if verbose:
            print(f"[RECONSTRUCTION_PROMPT] '{reconstruction_prompt}'")
            print(f"[CONDITIONING_PROMPT] '{conditioning_prompt}'")
            print(f"[DEPTHMAP_FROM_{'PROMPT' if inf_config['depthmap_from_prompt'] else 'SENSOR'}] B = {prompted_baseline}")
            print(f"[DEPTHMAP_FROM_{'PROMPT' if inf_config['depthmap_from_prompt'] else 'SENSOR'}] f = {focal_length}")

        left_gt = load_512(left_img_path)
        right_gt = load_512(sample["right.jpg"])
        disparity_gt = load_exr(sample["disparity.exr"])

        image_inv, latent, depth_to_disparity, disparity = run_inv_sd(
            left_img_path,
            prompts,
            camera_params,
            ldm_stable,
            depthmodel,
            disparitymodel,
            inf_config,
            cam_mlp=cam_mlp,
            verbose=verbose
        )

        left_gen, right_gen = image_inv

        # left
        this_psnr = psnr(left_gt, left_gen)
        this_ssim = ssim(left_gt, left_gen)
        this_lpips = lpips(left_gt, left_gen)

        sample_config["left_psnr"] = this_psnr
        all_left_psnr_scores.append(this_psnr)
        sample_config["left_ssim"] = this_ssim
        all_left_ssim_scores.append(this_ssim)
        sample_config["left_lpips"] = this_lpips
        all_left_lpips_scores.append(this_lpips)

        # right
        this_psnr = psnr(right_gt, right_gen)
        this_ssim = ssim(right_gt, right_gen)
        this_lpips = lpips(right_gt, right_gen)

        sample_config["right_psnr"] = this_psnr
        all_right_psnr_scores.append(this_psnr)
        sample_config["right_ssim"] = this_ssim
        all_right_ssim_scores.append(this_ssim)
        sample_config["right_lpips"] = this_lpips
        all_right_lpips_scores.append(this_lpips)

        # disparity
        this_lpips1 = lpips(disparity_gt, disparity)
        this_lpips2 = lpips(disparity_gt, depth_to_disparity)

        sample_config["disp_lpips"] = this_lpips1
        sample_config["depth-to-disp_lpips"] = this_lpips2
        all_disp_lpips_scores.append(this_lpips1)
        all_depth_to_disp_lpips_scores.append(this_lpips2)

        save_config(sample_config, f"{inf_config['output_prefix']}sample_config.json")

        inf_config["output_prefix"] = root_output_prefix # revert overridden output prefix
        counter += 1

    # save configs
    cfg_save_path = f"{inf_config['output_prefix']}cfg{os.sep}"
    os.makedirs(cfg_save_path, exist_ok=True)
    metrics = lambda x: [float(np.mean(x)), float(np.std(x))]
    eval_metrics_config = {
        "mean_std_psnr_left": metrics(all_left_psnr_scores),
        "mean_std_ssim_left": metrics(all_left_ssim_scores),
        "mean_std_lpips_left": metrics(all_left_lpips_scores),
        "mean_std_psnr_right": metrics(all_right_psnr_scores),
        "mean_std_ssim_right": metrics(all_right_ssim_scores),
        "mean_std_lpips_right": metrics(all_right_lpips_scores),
        "mean_std_lpips_deph_to_disp": metrics(all_depth_to_disp_lpips_scores),
        "mean_std_lpips_disp": metrics(all_disp_lpips_scores)
    }
    for t in [(inf_config, "inference_config.json"), (qpi_config, "qwen_config.json"), (eval_metrics_config, "eval_metrics_config.json")]:
        save_config(t[0], f"{cfg_save_path}{t[1]}")


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

sys.path.append('./DensePredictionTransformer')
from DensePredictionTransformer.dpt.models import DPTDepthModel
from stereoutils import stereo_shift_torch, norm_depth, BNAttention, register_attention_editor_diffusers, load_512

sys.path.append('./PromptToPrompt')
import ptp_utils
from ptp_null_text import AttentionStore, make_controller

sys.path.append('..')
from QwenPromptInterpreter.prompt2float import interpret_prompt
from misc_util import get_config, add_subfolder_to_save_prefix, save_config
from ptp_save_util import save_images, save_cross_attention, save_hist_from_array, save_generated_stereoimages
from stereodiffusion_nti import EmptyControl, NullInversion


def run_and_display(
    ldm_stable, 
    prompts, 
    controller, 
    disparity,
    inf_config,
    latent=None,
    run_baseline=False,
    generator=None, 
    uncond_embeddings=None,
    verbose=True,
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
            latent=torch.concat([latent,latent],0),
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
        uncond_embeddings=uncond_embeddings, 
        latent=latent,
        verbose=verbose
    )
    save_generated_stereoimages(images, inf_config["output_prefix"]) # unaffected by verbose, unlike every other save
    return images, latent


@torch.no_grad()
def text2stereoimage_ldm_stable(
    model,
    prompts: List[str],
    controller,
    disparity,
    inf_config,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
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

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)

    _latents_init = ptp_utils.latent2image(model.vae, latents)
    if verbose:
        save_images(_latents_init, f'{inf_config["output_prefix"]}initial_latents.png')

    model.scheduler.set_timesteps(inf_config["num_ddim_steps"])
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-inf_config["num_ddim_steps"]:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, inf_config["guidance_scale"], low_resource=inf_config["low_resource"])
        
        if (i % latents_editing_freq == 0) and verbose:
            _latents_at_t = ptp_utils.latent2image(model.vae, latents)
            save_images(_latents_at_t, f'{inf_config["output_prefix"]}latents_at_t={t}.png')
        
        # also reconstruct a right-side stereo image (StereoDiffusion)
        if i == latents_editing_freq:
            if isinstance(disparity,torch.Tensor):
                disparity = torch.nn.functional.interpolate(disparity.unsqueeze(1),size=[64,64],mode="bicubic",align_corners=False,).squeeze(1)
            elif isinstance(disparity,np.ndarray):
                disparity = resize(disparity,(64,64))
            
            scale_factor_percent = 8
            latents_current = stereo_shift_torch(
                latents[:1], # left latent
                disparity, 
                scale_factor_percent=scale_factor_percent
            )
            latents_current = latents_current[1:] # latents_current <- right latent
            latents = torch.cat([latents[:1], latents_current], 0) # [left latent, left latent shifted (right latent)]

            if verbose:
                _latents_at_t = ptp_utils.latent2image(model.vae, latents)
                save_images(_latents_at_t, f'{inf_config["output_prefix"]}latents-after-shift_at_t={t}.png')

            mask = latents_current[:,0,...] != 0
            mask = rearrange(mask,'b h w ->b () h w').repeat(1,4,1,1)
            noise = torch.randn_like(latents)

            if verbose:
                _mask = mask
                _mask = ptp_utils.latent2image(model.vae, _mask)
                save_images(_mask, f'{inf_config["output_prefix"]}denoising-mask.png')

            if inf_config["stereodiffusion_deblur"]:
                latents[1:][~mask] = noise[1:][~mask]
                latents[1:][mask] = latents_current[mask]

        if  (i > latents_editing_freq and i % latents_editing_freq == 0):
            latents_current = stereo_shift_torch(
                latents[:1], # left latent
                disparity, 
                scale_factor_percent=scale_factor_percent
            )
            latents_current = latents_current[1:] # latents_current <- right latent
            latents[1:][mask] = latents_current[mask] # prev right latent * mask <- curr right latent * mask
            
            if verbose:
                _latents_masked = ptp_utils.latent2image(model.vae, latents)
                save_images(_latents_masked, f'{inf_config["output_prefix"]}latents-with-applied-mask_at_t={t}.png')

        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def get_baseline_and_focal_length(img_path, baseline_prompt, inf_config, qpi_config, verbose=False):
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
    verbose=False
):
    image = load_512(img_path)
    null_inversion = NullInversion(ldm_stable, inf_config)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(
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
        run_baseline=False, # 1 => run with EmptyControl() first (no prompt conditioning)
        latent=x_t, 
        uncond_embeddings=uncond_embeddings,
        verbose=verbose
    )
    if verbose:
        print("saving attention score maps...", end="")
        save_cross_attention([prompts[1]], ldm_stable.tokenizer, controller, 16, ["up", "down"], f'{inf_config["output_prefix"]}attention.png')
        print("done")

    return image_inv, latent


if __name__ == "__main__":
    inf_config = get_config(path="../cfg/inference_config.json")
    qpi_config = get_config(path="../cfg/qwen_config.json")
    os.makedirs(inf_config["output_prefix"], exist_ok=True)

    verbose=True

    # from here per image
    img_path = "../../../resources/car_left.jpg"
    baseline_prompt = None

    prompted_baseline, focal_length = get_baseline_and_focal_length(
        img_path, 
        baseline_prompt, 
        inf_config,
        qpi_config,
        verbose=verbose
    )
    reconstruction_prompt = f"a sports car in a museum, captured by a stereo camera with baseline distance 0 and focal length {focal_length}"
    conditioning_prompt = f"a sports car in a museum, captured by a stereo camera with baseline distance {prompted_baseline} and focal length {focal_length}"
    prompts = [
        reconstruction_prompt,
        conditioning_prompt
    ]
    camera_params = [
        prompted_baseline,
        focal_length
    ]
    if verbose:
        print(f"[RECONSTRUCTION_PROMPT] '{reconstruction_prompt}'")
        print(f"[CONDITIONING_PROMPT] '{conditioning_prompt}'")
        print(f"[DEPTHMAP_FROM_{'PROMPT' if inf_config['depthmap_from_prompt'] else 'SENSOR'}] B = {prompted_baseline}")
        print(f"[DEPTHMAP_FROM_{'PROMPT' if inf_config['depthmap_from_prompt'] else 'SENSOR'}] f = {focal_length}")

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

    image_inv, latent = run_inv_sd(
        img_path,
        prompts,
        camera_params,
        ldm_stable,
        depthmodel,
        disparitymodel,
        inf_config,
        verbose=verbose
    )

    # save configs
    cfg_save_path = f"{inf_config['output_prefix']}cfg{os.sep}"
    os.makedirs(cfg_save_path, exist_ok=True)
    for t in [(inf_config, "inference_config.json"), (qpi_config, "qwen_config.json")]:
        save_config(t[0], f"{cfg_save_path}{t[1]}")


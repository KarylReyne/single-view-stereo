import argparse, os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import sys
from typing import Optional, Union, List
sys.path.append('./StableDiffusion')
sys.path.append('./DensePredictionTransformer')
from DensePredictionTransformer.dpt.models import DPTDepthModel
from stereoutils import stereo_shift_torch, norm_depth, BNAttention, register_attention_editor_diffusers, load_512
sys.path.append('./PromptToPrompt')
import ptp_utils
from ptp_null_text import AttentionStore, make_controller
from skimage.transform import resize
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
from torch.optim.adam import Adam

sys.path.append('..')
from QwenPromptInterpreter.prompt2float import interpret_prompt
from misc_util import get_config, create_save_path_from_prefix, add_subfolder_to_save_prefix
from ptp_save_util import save_images, save_cross_attention, save_hist_from_array
from stereodiffusion_nti import EmptyControl, NullInversion


def run_and_display(
    ldm_stable, 
    prompts, 
    controller, 
    disparity, 
    deblur, 
    latent=None, 
    run_baseline=False, 
    generator=None, 
    uncond_embeddings=None, 
    reconstruct_single_image=False, 
    verbose=True, 
    save_prefix=None
):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(
            ldm_stable, 
            ["", ""],
            EmptyControl(),
            disparity, 
            deblur, 
            latent=torch.concat([latent,latent],0),
            run_baseline=False, 
            generator=generator, 
            uncond_embeddings=uncond_embeddings,
            reconstruct_single_image=reconstruct_single_image,
            verbose=verbose,
            save_prefix=os.sep.join([save_prefix, "without-ptp"])
        )
        print("with prompt-to-prompt")
    images, latent = text2stereoimage_ldm_stable(
        ldm_stable, 
        prompts,
        controller,
        disparity, 
        uncond_embeddings=uncond_embeddings, 
        latent=latent,
        deblur=deblur,
        reconstruct_single_image=reconstruct_single_image,
        verbose=verbose,
        save_prefix=save_prefix
    )
    if verbose and (save_prefix != None):
        save_images(images, save_prefix+"_images_inference.png")
    return images, latent


@torch.no_grad()
def text2stereoimage_ldm_stable(
    model,
    prompts:  List[str],
    controller,
    disparity,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    latents_editing_freq=10,
    return_type='image',
    deblur=False,
    reconstruct_single_image=False,
    verbose=False,
    save_prefix=None
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
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)

    _latents_init = ptp_utils.latent2image(model.vae, latents)
    if verbose and (save_prefix != None):
        save_images(_latents_init, f'{save_prefix}_initial_latents.png')

    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=LOW_RESOURCE)
        
        if (i % latents_editing_freq == 0) and verbose and (save_prefix != None):
            _latents_at_t = ptp_utils.latent2image(model.vae, latents)
            save_images(_latents_at_t, f'{save_prefix}_latents_at_t={t}.png')
        
        # also reconstruct a right-side stereo image (StereoDiffusion)
        if not reconstruct_single_image:

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

                if verbose and (save_prefix != None):
                    _latents_at_t = ptp_utils.latent2image(model.vae, latents)
                    save_images(_latents_at_t, f'{save_prefix}_latents-after-shift_at_t={t}.png')

                mask = latents_current[:,0,...] != 0
                mask = rearrange(mask,'b h w ->b () h w').repeat(1,4,1,1)
                noise = torch.randn_like(latents)

                if verbose and (save_prefix != None):
                    _mask = mask
                    _mask = ptp_utils.latent2image(model.vae, _mask)
                    save_images(_mask, f'{save_prefix}_denoising-mask.png')

                if deblur: # avoid blurry
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
                
                if verbose and (save_prefix != None):
                    _latents_masked = ptp_utils.latent2image(model.vae, latents)
                    save_images(_latents_masked, f'{save_prefix}_latents-with-applied-mask_at_t={t}.png')

        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def get_baseline_and_focal_length(img_path, baseline_prompt, config):
    # custom baseline distance and focal length
    prompted_baseline = None
    focal_length = None

    # set baseline via prompt
    if config["depthmap_from_prompt"]:
        prompted_baseline, focal_length = interpret_prompt(baseline_prompt, config)
        if prompted_baseline == 0.0:
            print(f"[DEPTHMAP_FROM_PROMPT] baseline can`t be {prompted_baseline}! setting B=1e-8")
            prompted_baseline = 1e-8

    # testing depthmap generation from sensor data (blender)
    if config["depthmap_from_sensor"]:
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

    if config["depthmap_from_prompt"] or config["depthmap_from_sensor"]:
        print(f"[DEPTHMAP_FROM_{'PROMPT' if config["depthmap_from_prompt"] else 'SENSOR'}] B = {prompted_baseline}")
        print(f"[DEPTHMAP_FROM_{'PROMPT' if config["depthmap_from_prompt"] else 'SENSOR'}] f = {focal_length}")

    return prompted_baseline, focal_length


def estimate_disparity_from_gt(
        image_gt, 
        prompted_baseline, 
        focal_length,
        config,
        device, 
        verbose=False
    ):
    maps_folder = add_subfolder_to_save_prefix(config, "disparity_maps")

    net_w = net_h = 384

    depthmodel = DPTDepthModel(
        path=config["depthmodel_path"],
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        invert=args.estimate_only_depth
    ).to(device)

    image_gt_ = torch.tensor(np.expand_dims(image_gt/255,0).transpose(0,3,1,2)/255, device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = depthmodel.forward(image_gt_)
    
    # estimate disparity/depth
    if args.estimate_only_depth:
        assert focal_length != None and prompted_baseline != None
        depth = prediction
        disparity = (focal_length*prompted_baseline)/depth
        depth = norm_depth(depth)
        disparity = norm_depth(disparity)
    else:
        disparity = norm_depth(prediction)

    # print estimated disparity/depth
    if args.estimate_only_depth:
        disparity_and_depth = [disparity, depth]
        for i in range(len(disparity_and_depth)):
            map = disparity_and_depth[i]
            map = rearrange(map, 'c h w -> (c h) w')
            map = map.cpu().numpy()
            map = np.uint8(map*255)
            disparity_and_depth[i] = map
        if verbose:
            Image.fromarray(disparity_and_depth[0]).save(f'{maps_folder}_DPT-depth.png')
            save_hist_from_array(disparity_and_depth[0], f'{maps_folder}_DPT-depth_hist.png', title=r"Histogram of $Z_{DPT}$", color_idx=1)
            Image.fromarray(disparity_and_depth[1]).save(f'{maps_folder}_DPT-depth-to-disparity_B{prompted_baseline}_f{focal_length}.png')
            save_hist_from_array(disparity_and_depth[1], f'{maps_folder}_DPT-depth-to-disparity_hist_B{prompted_baseline}_f{focal_length}.png', title=r"Histogram of $D(B_{sensor},f_{sensor},Z_{DPT})$", color_idx=3)
    else:
        map = disparity
        map = rearrange(map, 'c h w -> (c h) w')
        map = map.cpu().numpy()
        map = np.uint8(map*255)
        if verbose:
            Image.fromarray(map).save(f'{maps_folder}_DPT-disparity.png')
            save_hist_from_array(map, f'{maps_folder}_DPT-disparity_hist.png', title=r"Histogram of $D_{DPT}$", color_idx=0)
    del depthmodel

    return disparity


def run_inv_sd(image, inf_config, qpi_config):
    device = torch.device("cuda")
    create_save_path_from_prefix(args.output_prefix)

    prompted_baseline, focal_length = get_baseline_and_focal_length(args, config=qpi_config)

    # TODO define the null-text inversion reconstruction prompt (left empty by StereoDiffusion)
    # reconstruction_prompt = ""
    # reconstruction_prompt = "a cat sitting next to a mirror"
    # reconstruction_prompt = f"a cat sitting next to a mirror, captured by a stereo camera with baseline distance 0 and focal length {focal_length}"
    reconstruction_prompt = f"a sports car in a museum, captured by a stereo camera with baseline distance 0 and focal length {focal_length}"
    print(f"[RECONSTRUCTION_PROMPT] '{reconstruction_prompt}'")

    null_inversion = NullInversion(ldm_stable)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(
        image, 
        reconstruction_prompt, 
        offsets=(0,0,200,0), 
        verbose=True
    )
    del null_inversion

    disparity = estimate_disparity_from_gt(image_gt, args, prompted_baseline, focal_length, device, verbose=True)

    print("testing null-text inversion for stereo image conditioning...")
    # conditioning_prompt = f"a cat sitting next to a mirror, captured by a stereo camera with baseline distance {prompted_baseline} and focal length {focal_length}"
    conditioning_prompt = f"a sports car in a museum, captured by a stereo camera with baseline distance {prompted_baseline} and focal length {focal_length}"
    prompts = [
        reconstruction_prompt,
        conditioning_prompt
    ]
    print(f"[CONDITIONING_PROMPT] '{conditioning_prompt}'")

    USE_NORMAL_ATTENTION = True # 0 => StereoDiffusion's uni-attention
    print(f"[USE_NORMAL_ATTENTION] '{USE_NORMAL_ATTENTION}'")

    if USE_NORMAL_ATTENTION:
        # attention editing for cat
        # stereo_cond_save_prefix = add_subfolder_to_save_prefix(args, f"conditioning{os.sep}stereo-attention")
        # cross_replace_steps = {'default_': .8,}
        # self_replace_steps = .5
        # blend_word = ((('cat',), ('cat',))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
        # eq_params = {"words": (f"{prompted_baseline}",), "values": (2,)} # amplify attention to the word "tiger" by *2 

        # controller = make_controller(
        #     prompts, 
        #     True, 
        #     cross_replace_steps, 
        #     self_replace_steps, 
        #     tokenizer, 
        #     device, 
        #     MAX_NUM_WORDS, 
        #     NUM_DDIM_STEPS, 
        #     blend_word,
        #     eq_params
        # )
        folders = f"conditioning{os.sep}stereo-attention"
        if args.deblur:
            folders += f"{os.sep}deblurred"
        stereo_cond_save_prefix = add_subfolder_to_save_prefix(args, folders)

        controller = AttentionStore(low_resource=LOW_RESOURCE)
    else:
        folders = f"conditioning{os.sep}stereo-bnattention"
        if args.deblur:
            folders += f"{os.sep}deblurred"
        stereo_cond_save_prefix = add_subfolder_to_save_prefix(args, folders)

        controller = BNAttention(start_step=4, total_steps=50, direction=args.direction)

    image_inv, latent = run_and_display(
        ldm_stable,
        prompts, 
        controller, 
        disparity, 
        args.deblur, 
        run_baseline=False, # 1 => run with EmptyControl() first (no prompt conditioning)
        latent=x_t, 
        uncond_embeddings=uncond_embeddings,
        reconstruct_single_image=False,
        verbose=True,
        save_prefix=stereo_cond_save_prefix
    )
    print("saving images...", end="")
    save_images([image_gt, image_enc, image_inv[0]], f'{stereo_cond_save_prefix}_images_gt-rec-inv.png')
    save_images([image_gt, image_enc, image_inv[1]], f'{stereo_cond_save_prefix}_images_gt-rec-cond.png')
    save_cross_attention([prompts[1]], tokenizer, controller, 16, ["up", "down"], f'{stereo_cond_save_prefix}_images_cond_cross-attention.png')
    print("done")

    image_pair = rearrange(image_inv,'b h w c->h (b w) c')
    if args.estimate_only_depth:
        Image.fromarray(image_pair).save(f'{args.output_prefix}_DPT-depth-to-disparity_B{prompted_baseline}_f{focal_length}_image_pair.png')
    else:
        Image.fromarray(image_pair).save(f'{args.output_prefix}_DPT-disparity_image_pair.png')
    return image, image_pair


if __name__ == "__main__":
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        clip_sample=False, 
        set_alpha_to_one=False,
        steps_offset=1
    )

    inf_config = get_config(path="../cfg/inference_config.json")
    qpi_config = get_config(path="../cfg/qwen_config.json")

    img_path = None
    baseline_prompt = None

    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        scheduler=scheduler
    ).to(inf_config["device"])
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    image  = load_512(img_path)
    out_image, image_pair = run_inv_sd(image, inf_config, qpi_config)


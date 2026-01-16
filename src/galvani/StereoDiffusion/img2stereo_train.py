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
from stereoutils import stereo_shift_torch, norm_depth, BNAttention, register_attention_editor_diffusers
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


class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        # bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in tqdm(range(NUM_DDIM_STEPS)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            # for j in range(j + 1, num_inner_steps):
                # bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        # bar.close()
        return uncond_embeddings_list
    
    def invert(self, image, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        if isinstance(image, str):
            image_gt = load_512(image, *offsets)
        elif isinstance(image, np.ndarray):
            image_gt = resize(image, (512, 512))
            if image_gt.max()<=1:
                image_gt = (image_gt * 255).astype(np.uint8)
        else:
            raise ValueError("image_path must be either a path to an image or a numpy array")
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def run_and_display(ldm_stable, prompts, controller, disparity, deblur, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, reconstruct_single_image=False, verbose=True, save_prefix=None):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="path to image")
    parser.add_argument("--depthmodel_path",type=str,required=True, help='path of depth model')
    parser.add_argument("--output_prefix", type=str, required=True, help="prefix for saving the output")
    parser.add_argument(
        "--meta_path", 
        type=str, 
        help="path to metadata file"
    )
    parser.add_argument(
        "--estimate_only_depth", 
        action="store_true", 
        default=False, 
        help="whether DPT should estimate depth or disparity"
    )
    parser.add_argument(
        "--deblur",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=9.0,
        help="scale factor of disparity",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["uni", "bi"],
        default="uni"
    )
    parser.add_argument(
        "--baseline_prompt",
        type=str,
        default="default"
    )
    return parser.parse_args()


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    if w != h:
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


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


def get_baseline_and_focal_length(args):
    # custom baseline distance and focal length
    DEPTHMAP_FROM_PROMPT = False
    DEPTHMAP_FROM_SENSOR = True
    if args.baseline_prompt != "default":
        DEPTHMAP_FROM_PROMPT = True
        DEPTHMAP_FROM_SENSOR = False
    prompted_baseline = None
    focal_length = None

    # set baseline via prompt
    if DEPTHMAP_FROM_PROMPT:
        qpi_config = get_config(path="../QwenPromptInterpreter/cfg/config.json")
        prompted_baseline, focal_length = interpret_prompt(args.baseline_prompt, qpi_config)
        if prompted_baseline == 0.0:
            print(f"[DEPTHMAP_FROM_PROMPT] baseline can`t be {prompted_baseline}! setting B=1e-8")
            prompted_baseline = 1e-8

    # testing depthmap generation from sensor data (blender)
    if DEPTHMAP_FROM_SENSOR:
        # import metadata file
        if args.meta_path: # for everything NOT called "meta.json"
            metadata_path = args.meta_path
        else:
            metadata_path = args.img_path.split("/")[:-1]
            metadata_path.append("meta.json")
            metadata_path = "/".join(metadata_path)
        metadata = get_config(path=metadata_path)
        prompted_baseline = metadata["baseline_m"]
        focal_length = metadata["focal_mm"]

    if DEPTHMAP_FROM_PROMPT or DEPTHMAP_FROM_SENSOR:
        print(f"[DEPTHMAP_FROM_{'PROMPT' if DEPTHMAP_FROM_PROMPT else 'SENSOR'}] B = {prompted_baseline}")
        print(f"[DEPTHMAP_FROM_{'PROMPT' if DEPTHMAP_FROM_PROMPT else 'SENSOR'}] f = {focal_length}")

    return prompted_baseline, focal_length


def estimate_disparity_from_gt(
        image_gt, 
        args, 
        prompted_baseline, 
        focal_length,
        device, 
        verbose=False
    ):
    net_w = net_h = 384

    depthmodel = DPTDepthModel(
        path=args.depthmodel_path,
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
            Image.fromarray(disparity_and_depth[0]).save(f'{args.output_prefix}_DPT-depth.png')
            save_hist_from_array(disparity_and_depth[0], f'{args.output_prefix}_DPT-depth_hist.png', title=r"Histogram of $Z_{DPT}$", color_idx=1)
            Image.fromarray(disparity_and_depth[1]).save(f'{args.output_prefix}_DPT-depth-to-disparity_B{prompted_baseline}_f{focal_length}.png')
            save_hist_from_array(disparity_and_depth[1], f'{args.output_prefix}_DPT-depth-to-disparity_hist_B{prompted_baseline}_f{focal_length}.png', title=r"Histogram of $D(B_{sensor},f_{sensor},Z_{DPT})$", color_idx=3)
    else:
        map = disparity
        map = rearrange(map, 'c h w -> (c h) w')
        map = map.cpu().numpy()
        map = np.uint8(map*255)
        if verbose:
            Image.fromarray(map).save(f'{args.output_prefix}_DPT-disparity.png')
            save_hist_from_array(map, f'{args.output_prefix}_DPT-disparity_hist.png', title=r"Histogram of $D_{DPT}$", color_idx=0)
    del depthmodel

    return disparity


def run_inv_sd(image, args):
    device = torch.device("cuda")
    create_save_path_from_prefix(args.output_prefix)

    prompted_baseline, focal_length = get_baseline_and_focal_length(args)

    # TODO define the null-text inversion reconstruction prompt (left empty by StereoDiffusion)
    # reconstruction_prompt = ""
    # reconstruction_prompt = "a cat sitting next to a mirror"
    reconstruction_prompt = f"a cat sitting next to a mirror, captured by a stereo camera with baseline distance 0 and focal length {focal_length}"
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

    # print("testing null-text inversion reconstruction...")
    # # rec_save_prefix = add_subfolder_to_save_prefix(args, f"reconstruction{os.sep}left-no-prompt")
    # rec_save_prefix = add_subfolder_to_save_prefix(args, f"reconstruction{os.sep}left")
    # prompts = [
    #     reconstruction_prompt
    # ]
    # controller = AttentionStore(low_resource=LOW_RESOURCE)
    # image_inv, latent = run_and_display(
    #     ldm_stable,
    #     prompts, 
    #     controller, 
    #     disparity, 
    #     args.deblur, 
    #     run_baseline=False, # 1 => run with EmptyControl() first (no prompt conditioning)
    #     latent=x_t, 
    #     uncond_embeddings=uncond_embeddings,
    #     reconstruct_single_image=True, # 1 => run only Prompt-To-Prompt (only 'left' image reconstruction)
    #     verbose=False,
    #     save_prefix=rec_save_prefix'
    # )
    # print("saving images...", end="")
    # save_images([image_gt, image_enc, image_inv[0]], f'{rec_save_prefix}_images_gt-rec-inv.png')
    # save_cross_attention(prompts, tokenizer, controller, 16, ["up", "down"], f'{rec_save_prefix}_images_cross-attention.png')
    # print("done")

    # print("testing null-text inversion conditioning...")
    # cond_save_prefix = add_subfolder_to_save_prefix(args, f"conditioning{os.sep}stereo")
    # conditioning_prompt = "a tiger sitting next to a mirror"
    # print(f"[CONDITIONING_PROMPT] '{conditioning_prompt}'")
    # prompts = [
    #     reconstruction_prompt,
    #     conditioning_prompt
    # ]
    
    # cross_replace_steps = {'default_': .8,}
    # self_replace_steps = .5
    # blend_word = ((('cat',), ("tiger",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    # eq_params = {"words": ("tiger",), "values": (2,)} # amplify attention to the word "tiger" by *2 

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
    # image_inv, latent = run_and_display(
    #     ldm_stable,
    #     prompts, 
    #     controller, 
    #     disparity, 
    #     args.deblur, 
    #     run_baseline=False, # 1 => run with EmptyControl() first (no prompt conditioning)
    #     latent=x_t, 
    #     uncond_embeddings=uncond_embeddings,
    #     reconstruct_single_image=True,
    #     verbose=True,
    #     save_prefix=cond_save_prefix
    # )
    # print("saving images...", end="")
    # save_images([image_gt, image_enc, image_inv[0]], f'{cond_save_prefix}_images_gt-rec-inv.png')
    # save_images([image_gt, image_enc, image_inv[1]], f'{cond_save_prefix}_images_gt-rec-cond.png')
    # save_cross_attention([prompts[1]], tokenizer, controller, 16, ["up", "down"], f'{cond_save_prefix}_images_cond_cross-attention.png')
    # print("done")

    print("testing null-text inversion for stereo image conditioning...")
    USE_NORMAL_ATTENTION = False
    conditioning_prompt = f"a cat sitting next to a mirror, captured by a stereo camera with baseline distance {prompted_baseline} and focal length {focal_length}"
    print(f"[CONDITIONING_PROMPT] '{conditioning_prompt}'")
    prompts = [
        reconstruction_prompt,
        conditioning_prompt
    ]
    if USE_NORMAL_ATTENTION:
        stereo_cond_save_prefix = add_subfolder_to_save_prefix(args, f"conditioning{os.sep}stereo-attention")
        cross_replace_steps = {'default_': .8,}
        self_replace_steps = .5
        blend_word = ((('cat',), ('cat',))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
        eq_params = {"words": (f"{prompted_baseline}",), "values": (2,)} # amplify attention to the word "tiger" by *2 

        controller = make_controller(
            prompts, 
            True, 
            cross_replace_steps, 
            self_replace_steps, 
            tokenizer, 
            device, 
            MAX_NUM_WORDS, 
            NUM_DDIM_STEPS, 
            blend_word, 
            eq_params
        )
    else:
        stereo_cond_save_prefix = add_subfolder_to_save_prefix(args, f"conditioning{os.sep}stereo-bnattention")
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
    args = parse_args()

    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        clip_sample=False, 
        set_alpha_to_one=False,
        steps_offset=1
    )
    device = "cuda:0"
    GUIDANCE_SCALE = 7.5
    NUM_DDIM_STEPS = 50
    MAX_NUM_WORDS = 77
    LOW_RESOURCE = False
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        scheduler=scheduler
    ).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    image  = load_512(args.img_path)
    out_image, image_pair = run_inv_sd(image, args)


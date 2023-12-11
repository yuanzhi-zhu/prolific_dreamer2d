import os
join = os.path.join
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
import random
from model_utils import (
            get_t_schedule, 
            loss_weights, 
            sds_vsd_grad_diffuser, 
            phi_vsd_grad_diffuser, 
            extract_lora_diffusers,
            predict_noise0_diffuser,
            update_curve
)
import shutil
import logging

# from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler

IMG_EXTENSIONS = ['jpg', 'png', 'jpeg', 'bmp']

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # parameters
    ### basics
    parser.add_argument('--seed', default=1024, type=int, help='global seed')
    parser.add_argument('--log_steps', type=int, default=50, help='Log steps')
    parser.add_argument('--log_progress', type=str2bool, default=False, help='Log progress')
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4', help='Path to the model')
    current_datetime = datetime.now()
    parser.add_argument('--run_date', type=str, default=current_datetime.strftime("%Y%m%d"), help='Run date')
    parser.add_argument('--run_time', type=str, default=current_datetime.strftime("%H%M"), help='Run time')
    parser.add_argument('--work_dir', type=str, default='work_dir/prolific_dreamer2d', help='Working directory')
    parser.add_argument('--half_inference', type=str2bool, default=False, help='inference sd with half precision')
    parser.add_argument('--save_x0', type=str2bool, default=False, help='save predicted x0')
    parser.add_argument('--save_phi_model', type=str2bool, default=False, help='save save_phi_model, lora or simple unet')
    parser.add_argument('--load_phi_model_path', type=str, default='', help='phi_model_path to load')
    parser.add_argument('--use_mlp_particle', type=str2bool, default=False, help='use_mlp_particle as representation')
    ### sampling
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps for random sampling')
    parser.add_argument('--t_end', type=int, default=980, help='largest possible timestep for random sampling')
    parser.add_argument('--t_start', type=int, default=20, help='least possible timestep for random sampling')
    parser.add_argument('--multisteps', default=1, type=int, help='multisteps to predict x0')
    parser.add_argument('--t_schedule', default='descend', type=str, help='t_schedule for sampling')
    parser.add_argument('--prompt', default="a photograph of an astronaut riding a horse", type=str, help='prompt')
    parser.add_argument('--height', default=512, type=int, help='height of image')
    parser.add_argument('--width', default=512, type=int, help='width of image')
    parser.add_argument('--rgb_as_latents', default=True, type=str2bool, help='width of image')
    parser.add_argument('--generation_mode', default='sds', type=str, help='sd generation mode')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size / overall number of particles')
    parser.add_argument('--particle_num_vsd', default=1, type=int, help='batch size for VSD training')
    parser.add_argument('--particle_num_phi', default=1, type=int, help='number of particles to train phi model')
    parser.add_argument('--guidance_scale', default=7.5, type=float, help='Scale for classifier-free guidance')
    parser.add_argument('--cfg_phi', default=1., type=float, help='Scale for classifier-free guidance of phi model')
    ### optimizing
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--phi_lr', type=float, default=0.0001, help='Learning rate for phi model')
    parser.add_argument('--phi_model', type=str, default='lora', help='models servered as epsilon_phi')
    parser.add_argument('--use_t_phi', type=str2bool, default=False, help='use different t for phi finetuning')
    parser.add_argument('--phi_update_step', type=int, default=1, help='phi finetuning steps in each iteration')
    parser.add_argument('--lora_vprediction', type=str2bool, default=False, help='use v prediction model for lora')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='lora_scale of the unet cross attn')
    parser.add_argument('--use_scheduler', default=False, type=str2bool, help='use_scheduler for lr')
    parser.add_argument('--lr_scheduler_start_factor', type=float, default=1/3, help='Start factor for learning rate scheduler')
    parser.add_argument('--lr_scheduler_iters', type=int, default=300, help='Iterations for learning rate scheduler')
    parser.add_argument('--loss_weight_type', type=str, default='none', help='type of loss weight')
    parser.add_argument('--nerf_init', type=str2bool, default=False, help='initialize with diffusion models as mean predictor')
    parser.add_argument('--grad_scale', type=float, default=1., help='grad_scale for loss in vsd')
    parser.add_argument('--grad_scale_phi', type=float, default=1., help='grad_scale for phi loss in vsd')
    args = parser.parse_args()
    # create working directory
    args.run_id = args.run_date + '_' + args.run_time
    args.work_dir = f'{args.work_dir}_{args.run_id}_{args.generation_mode}_cfg_{args.guidance_scale}_bs_{args.batch_size}_num_steps_{args.num_steps}_tschedule_{args.t_schedule}'
    args.work_dir = args.work_dir + f'_{args.phi_model}' if args.generation_mode == 'vsd' else args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)
    assert args.generation_mode in ['t2i', 'sds', 'vsd']
    assert args.phi_model in ['lora', 'unet_simple', 'img_variant', 'instruct_p2p']
    if args.half_inference:
        assert args.generation_mode in ['t2i', 'sds'], "half precision doesnot support vsd"
    # for sds and t2i, use only args.batch_size
    if args.generation_mode in ['t2i', 'sds']:
        args.particle_num_vsd = args.batch_size
        args.particle_num_phi = args.batch_size
    assert (args.batch_size >= args.particle_num_vsd) and (args.batch_size >= args.particle_num_phi)
    if args.batch_size > args.particle_num_vsd:
        print(f'use multiple ({args.batch_size}) particles!! Will get inconsistent x0 recorded')
    ### set random seed everywhere
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    return args


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    

def main():
    args = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # use float32 by default
    image_name = args.prompt.replace(' ', '_')
    shutil.copyfile(__file__, join(args.work_dir, os.path.basename(__file__)))
    ### set up logger
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.basicConfig(filename=f'{args.work_dir}/std_{args.run_id}.log', filemode='w', 
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(f'[INFO] Cmdline: '+' '.join(sys.argv))
    ### log basic info
    args.device = device
    logger.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(0))
    logger.info("################# Arguments: ####################")
    for arg in vars(args):
        logger.info(f"\t{arg}: {getattr(args, arg)}")
    
    #######################################################################################
    ### load model
    logger.info(f'load models from path: {args.model_path}')
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype)
    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype)
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet", torch_dtype=dtype)
    # 4. Scheduler
    scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler", torch_dtype=dtype)

    if args.half_inference:
        unet = unet.half()
        vae = vae.half()
        text_encoder = text_encoder.half()
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # all variables in same device for scheduler.step()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    if args.generation_mode == 'vsd':
        if args.phi_model == 'lora':
            if args.lora_vprediction:
                assert args.model_path == 'stabilityai/stable-diffusion-2-1-base'
                phi_model_path = 'stabilityai/stable-diffusion-2-1'
                tokenizer_phi = CLIPTokenizer.from_pretrained(phi_model_path, subfolder="tokenizer", torch_dtype=dtype)
                text_encoder_phi = CLIPTextModel.from_pretrained(phi_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
                vae_phi = AutoencoderKL.from_pretrained(phi_model_path, subfolder="vae", torch_dtype=dtype).to(device)
                unet_phi = UNet2DConditionModel.from_pretrained(phi_model_path, subfolder="unet", torch_dtype=dtype).to(device)
                vae_phi.requires_grad_(False)
                unet_phi, unet_lora_layers = extract_lora_diffusers(unet_phi, device)
            else:
                tokenizer_phi = tokenizer
                text_encoder_phi = text_encoder
                vae_phi = vae
                ### unet_phi is the same instance as unet that has been modified in-place
                unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            if args.load_phi_model_path:
                unet_phi.load_attn_procs(args.load_phi_model_path)
                unet_phi = unet_phi.to(device)
        elif args.phi_model == 'img_variant':
            ## set up phi model as img_variant instead of lora
            from torchvision import transforms
            args.save_phi_model = False
            args.cfg_phi = 3.
            ### https://huggingface.co/lambdalabs/sd-image-variations-diffusers
            tokenizer_phi = tokenizer
            text_encoder_phi = text_encoder
            phi_model_path = 'lambdalabs/sd-image-variations-diffusers'
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(phi_model_path, revision="v2.0", subfolder="image_encoder", torch_dtype=dtype).to(device)
            vae_phi = AutoencoderKL.from_pretrained(phi_model_path, revision="v2.0", subfolder="vae", torch_dtype=dtype).to(device)
            unet_phi = UNet2DConditionModel.from_pretrained(phi_model_path, revision="v2.0", subfolder="unet", torch_dtype=dtype).to(device)
            image_encoder.requires_grad_(False)
            vae_phi.requires_grad_(False)
            unet_phi.requires_grad_(False)
            tform = transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                    ),
                transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
            ])
        elif args.phi_model == 'instruct_p2p':
            ## set up phi model as instruct_p2p instead of lora
            from torchvision import transforms
            args.save_phi_model = False
            args.cfg_phi = 7.5
            ### https://huggingface.co/lambdalabs/sd-image-variations-diffusers
            phi_model_path = 'timbrooks/instruct-pix2pix'
            tokenizer_phi = CLIPTokenizer.from_pretrained(phi_model_path, subfolder="tokenizer", torch_dtype=dtype)
            text_encoder_phi = CLIPTextModel.from_pretrained(phi_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
            vae_phi = AutoencoderKL.from_pretrained(phi_model_path, subfolder="vae", torch_dtype=dtype).to(device)
            unet_phi = UNet2DConditionModel.from_pretrained(phi_model_path, subfolder="unet", torch_dtype=dtype).to(device)
            text_encoder_phi.requires_grad_(False)
            vae_phi.requires_grad_(False)
            unet_phi.requires_grad_(False)
        elif args.phi_model == 'unet_simple':
            # initialize simple unet, same input/output as (pre-trained) unet
            ### IMPORTANT: need the proper (wide) channel numbers
            channels = 4 if args.rgb_as_latents else 3
            tokenizer_phi = tokenizer
            text_encoder_phi = text_encoder
            unet_phi = UNet2DConditionModel(
                                        sample_size=64,
                                        in_channels=channels,
                                        out_channels=channels,
                                        layers_per_block=1,
                                        block_out_channels=(64,128,256),
                                        down_block_types=(
                                            "CrossAttnDownBlock2D",
                                            "CrossAttnDownBlock2D",
                                            "DownBlock2D",
                                        ),
                                        up_block_types=(
                                            "UpBlock2D",
                                            "CrossAttnUpBlock2D",
                                            "CrossAttnUpBlock2D",
                                        ),
                                        cross_attention_dim=unet.config.cross_attention_dim,
                                        ).to(dtype)
            if args.load_phi_model_path:
                unet_phi = unet_phi.from_pretrained(args.load_phi_model_path)
            unet_phi = unet_phi.to(device)
            phi_params = list(unet_phi.parameters())
            vae_phi = vae
    elif args.generation_mode == 'sds':
        unet_phi = None
    
    ### get text embedding
    text_input = tokenizer([args.prompt] * args.particle_num_vsd, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * args.particle_num_vsd, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings_vsd = torch.cat([uncond_embeddings, text_embeddings])
    if args.generation_mode == 'vsd':
        text_input_phi = tokenizer_phi([args.prompt] * args.particle_num_phi, padding="max_length", max_length=tokenizer_phi.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings_phi = text_encoder_phi(text_input_phi.input_ids.to(device))[0]
        max_length = text_input_phi.input_ids.shape[-1]
        uncond_input_phi = tokenizer_phi(
            [""] * args.particle_num_phi, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings_phi = text_encoder_phi(uncond_input_phi.input_ids.to(device))[0]
        text_embeddings_phi = torch.cat([uncond_embeddings_phi, text_embeddings_phi])
    
    ### weight loss
    num_train_timesteps = len(scheduler.betas)
    loss_weight = loss_weights(scheduler.betas, args)

    ### scheduler set timesteps
    if args.generation_mode == 't2i':
        scheduler.set_timesteps(args.num_steps)
    else:
        scheduler.set_timesteps(num_train_timesteps)

    ### initialize particles
    if args.use_mlp_particle:
        # use siren network
        from model_utils import Siren
        print(f'for mlp_particle, set lr to 1e-4')
        args.lr = 1e-4
        out_features = 4 if args.rgb_as_latents else 3
        particles = nn.ModuleList([Siren(2, hidden_features=256, hidden_layers=3, out_features=out_features, device=device) for _ in range(args.batch_size)])
    else:
        if args.rgb_as_latents:
            particles = torch.randn((args.batch_size, unet.config.in_channels, args.height // 8, args.width // 8))
        else:
            # gaussian in rgb space --> strange artifacts
            particles = torch.randn((args.batch_size, 3, args.height, args.width))
            args.lr = args.lr * 1   # need larger lr for rgb particles
            # ## gaussian in latent space --> not better
            # particles = torch.randn((args.batch_size, unet.in_channels, args.height // 8, args.width // 8)).to(device, dtype=dtype)
            # particles = vae.decode(particles).sample
    particles = particles.to(device, dtype=dtype)
    if args.nerf_init and args.rgb_as_latents and not args.use_mlp_particle:
        # current only support sds and experimental for only rgb_as_latents==True
        assert args.generation_mode == 'sds'
        with torch.no_grad():
            noise_pred = predict_noise0_diffuser(unet, particles, text_embeddings_vsd, t=999, guidance_scale=7.5, scheduler=scheduler)
        particles = scheduler.step(noise_pred, 999, particles).pred_original_sample
    # latents = latents * scheduler.init_noise_sigma
    
    def get_latents(particles, rgb_as_latents=False, use_mlp_particle=False):
        ### get latents from particles
        if use_mlp_particle:
            images = []
            output_size = args.height // 8 if rgb_as_latents else args.height
            # Loop over all MLPs and generate an image for each
            for particle_mlp in particles:
                image = particle_mlp.generate_image(output_size)
                images.append(image)
            # Stack all images together
            latents = torch.cat(images, dim=0)
            if not rgb_as_latents:
                latents = 0.18215 * vae.encode(latents).latent_dist.sample()
        else:
            if rgb_as_latents:
                latents = F.interpolate(particles, (64, 64), mode="bilinear", align_corners=False)
            else:
                rgb_BCHW_512 = F.interpolate(particles, (512, 512), mode="bilinear", align_corners=False)
                # encode image into latents with vae
                latents = 0.18215 * vae.encode(rgb_BCHW_512).latent_dist.sample()
        return latents

    #######################################################################################
    ### configure optimizer and loss function
    if args.use_mlp_particle:
        # For a list of models, we want to optimize their parameters
        particles_to_optimize = [param for mlp in particles for param in mlp.parameters() if param.requires_grad]
    else:
        # For a tensor, we can optimize the tensor directly
        particles.requires_grad = True
        particles_to_optimize = [particles]

    total_parameters = sum(p.numel() for p in particles_to_optimize if p.requires_grad)
    print(f'Total number of trainable parameters in particles: {total_parameters}; number of particles: {args.batch_size}')
    ### Initialize optimizer & scheduler
    if args.generation_mode == 'vsd':
        if args.phi_model in ['lora', 'unet_simple']:
            phi_optimizer = torch.optim.AdamW([{"params": phi_params, "lr": args.phi_lr}], lr=args.phi_lr)
            print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')
    optimizer = torch.optim.Adam(particles_to_optimize, lr=args.lr)
    if args.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, \
            start_factor=args.lr_scheduler_start_factor, total_iters=args.lr_scheduler_iters)

    #######################################################################################
    ############################# Main optimization loop ##############################
    log_steps = []
    train_loss_values = []
    ave_train_loss_values = []
    if args.log_progress:
        image_progress = []
    first_iteration = True
    logger.info("################# Metrics: ####################")
    ######## t schedule #########
    chosen_ts = get_t_schedule(num_train_timesteps, args, loss_weight)
    pbar = tqdm(chosen_ts)
    ### regular sd text to image generation
    image_latents = None
    if args.generation_mode == 't2i':
        if args.phi_model == 'lora' and args.load_phi_model_path:
            ### unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            unet_phi.load_attn_procs(args.load_phi_model_path)
            unet = unet_phi.to(device)
        step = 0
        # get latent of all particles
        assert args.use_mlp_particle == False
        latents = get_latents(particles, args.rgb_as_latents)
        if args.half_inference:
            latents = latents.half()
            text_embeddings_vsd = text_embeddings_vsd.half()
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_noisy = latents
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_vsd).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            ######## Evaluation and log metric #########
            if args.log_steps and (step % args.log_steps == 0 or step == (args.num_steps-1)):
                # save current img_tensor
                # scale and decode the image latents with vae
                tmp_latents = 1 / 0.18215 * latents.clone().detach()
                if args.save_x0:
                    # compute the predicted clean sample x_0
                    pred_latents = scheduler.step(noise_pred, t, latent_noisy).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    if args.half_inference:
                        tmp_latents = tmp_latents.half()
                    image_ = vae.decode(tmp_latents).sample.to(torch.float32)
                    if args.save_x0:
                        if args.half_inference:
                            pred_latents = pred_latents.half()
                        image_x0 = vae.decode(pred_latents / 0.18215).sample.to(torch.float32)
                        image = torch.cat((image_,image_x0), dim=2)
                if args.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
            step += 1
    ### sds text to image generation
    elif args.generation_mode in ['sds', 'vsd']:
        text_embeddings_phi = None
        cross_attention_kwargs = {'scale': args.lora_scale} if (args.generation_mode == 'vsd' and args.phi_model == 'lora') else {}
        for step, chosen_t in enumerate(pbar):
            # get latent of all particles
            latents = get_latents(particles, args.rgb_as_latents, use_mlp_particle=args.use_mlp_particle)
            t = torch.tensor([chosen_t]).to(device)
            ######## q sample #########
            # random sample particle_num_vsd particles from latents
            indices = torch.randperm(latents.size(0))
            latents_vsd = latents[indices[:args.particle_num_vsd]]
            noise = torch.randn_like(latents_vsd)
            noisy_latents = scheduler.add_noise(latents_vsd, noise, t)
            ######## Do the gradient for latents!!! #########
            optimizer.zero_grad()
            # predict x0 use ddim sampling
            # z0_latents = predict_x0_diffuser(unet, scheduler, noisy_latents, text_embeddings, t, guidance_scale=args.guidance_scale)
            # loss step
            if args.generation_mode == 'vsd' and args.phi_model in ['img_variant', 'instruct_p2p']:
                tmp_latents = 1 / 0.18215 * latents_vsd.clone().detach()
                image_vsd = vae.decode(tmp_latents).sample.to(torch.float32)
                assert image_vsd.shape[0] == 1
                if args.phi_model == 'img_variant':
                    image_vsd = tform(image_vsd).to(device)
                    ## https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_image_variation.py#L162
                    ## output is too diverse
                    img_emb = image_encoder(image_vsd).image_embeds
                    img_emb = img_emb.unsqueeze(1)
                    negative_prompt_embeds = torch.zeros_like(img_emb)
                    # CFG embedding of the image, but still in the name of text_embeddings
                    text_embeddings_phi = torch.cat([negative_prompt_embeds, img_emb])  
                elif args.phi_model == 'instruct_p2p':
                    ## https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#361
                    image_latents = vae_phi.encode(image_vsd).latent_dist.mode()
                    uncond_image_latents = torch.zeros_like(image_latents)
                    image_latents = torch.cat([uncond_image_latents, image_latents, image_latents], dim=0)

            loss, noise_pred, noise_pred_phi = sds_vsd_grad_diffuser(unet, noisy_latents, noise, text_embeddings_vsd, t, \
                                                    guidance_scale=args.guidance_scale, unet_phi=unet_phi, \
                                                        generation_mode=args.generation_mode, phi_model=args.phi_model, \
                                                            cross_attention_kwargs=cross_attention_kwargs, \
                                                                multisteps=args.multisteps, scheduler=scheduler, lora_v=args.lora_vprediction, \
                                                                    half_inference=args.half_inference, grad_scale_phi=args.grad_scale_phi, \
                                                                        cfg_phi=args.cfg_phi, grad_scale=args.grad_scale, \
                                                                            text_embeddings_phi=text_embeddings_phi, \
                                                                                image_latents=image_latents)
            ## weighting
            loss *= loss_weight[int(t)]
            ## Compute gradients
            loss.backward()
            optimizer.step()
            if args.use_scheduler:
                lr_scheduler.step(loss)

            torch.cuda.empty_cache()
            ######## Do the gradient for unet_phi!!! #########
            if args.generation_mode == 'vsd' and args.phi_model not in ['img_variant', 'instruct_p2p']:
                ## update the unet (phi) model 
                for _ in range(args.phi_update_step):
                    phi_optimizer.zero_grad()
                    if args.use_t_phi:
                        # different t for phi finetuning
                        # t_phi = np.random.choice(chosen_ts, 1, replace=True)[0]
                        t_phi = np.random.choice(list(range(num_train_timesteps)), 1, replace=True)[0]
                        t_phi = torch.tensor([t_phi]).to(device)
                    else:
                        t_phi = t
                    # random sample particle_num_phi particles from latents
                    indices = torch.randperm(latents.size(0))
                    latents_phi = latents[indices[:args.particle_num_phi]]
                    noise_phi = torch.randn_like(latents_phi)
                    noisy_latents_phi = scheduler.add_noise(latents_phi, noise_phi, t_phi)
                    loss_phi = phi_vsd_grad_diffuser(unet_phi, noisy_latents_phi.detach(), noise_phi, text_embeddings_phi, t_phi, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, lora_v=args.lora_vprediction)
                    loss_phi.backward()
                    phi_optimizer.step()

            ### Store loss and step
            train_loss_values.append(loss.item())
            ### update pbar
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {t.item()}')

            optimizer.zero_grad()
            ######## Evaluation and log metric #########
            if args.log_steps and (step % args.log_steps == 0 or step == (args.num_steps-1)):
                log_steps.append(step)
                # save current img_tensor
                # scale and decode the image latents with vae
                tmp_latents = 1 / 0.18215 * latents_vsd.clone().detach()
                if args.save_x0:
                    # compute the predicted clean sample x_0
                    # pred_latents = scheduler.step(noise_pred, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                    pred_latents = scheduler.step(noise_pred-noise_pred_phi+noise, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                    if args.generation_mode == 'vsd':
                        pred_latents_phi = scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    if args.half_inference:
                        tmp_latents = tmp_latents.half()
                    image_ = vae.decode(tmp_latents).sample.to(torch.float32)
                    if args.save_x0:
                        if args.half_inference:
                            pred_latents = pred_latents.half()
                        image_x0 = vae.decode(pred_latents / 0.18215).sample.to(torch.float32)
                        if args.generation_mode == 'vsd':
                            image_x0_phi = vae_phi.decode(pred_latents_phi / 0.18215).sample.to(torch.float32)
                            image = torch.cat((image_,image_x0,image_x0_phi), dim=2)
                        else:
                            image = torch.cat((image_,image_x0), dim=2)
                if args.log_progress:
                    image_progress.append((torch.cat((image_,image_x0), dim=2)/2+0.5).clamp(0, 1))
                save_image((image/2+0.5).clamp(0, 1), f'{args.work_dir}/{image_name}_image_step{step}_t{t.item()}.png')
                ave_train_loss_value = np.average(train_loss_values)
                ave_train_loss_values.append(ave_train_loss_value) if step > 0 else None
                logger.info(f'step: {step}; average loss: {ave_train_loss_value}')
                update_curve(train_loss_values, 'Train_loss', 'steps', 'Loss', args.work_dir, args.run_id)
                update_curve(ave_train_loss_values, 'Ave_Train_loss', 'steps', 'Loss', args.work_dir, args.run_id, log_steps=log_steps[1:])
                # calculate psnr value and update curve
            if first_iteration and device==torch.device('cuda'):
                global_free, total_gpu = torch.cuda.mem_get_info(0)
                logger.info(f'global free and total GPU memory: {round(global_free/1024**3,6)} GB, {round(total_gpu/1024**3,6)} GB')
                first_iteration = False

    if args.log_progress and args.batch_size == 1:
        concatenated_images = torch.cat(image_progress, dim=0)
        save_image(concatenated_images, f'{args.work_dir}/{image_name}_prgressive.png')
    with torch.no_grad():
        # get latent of all particles
        latents = get_latents(particles, args.rgb_as_latents, use_mlp_particle=args.use_mlp_particle)
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents.clone()
    torch.cuda.empty_cache()
    if args.generation_mode == 't2i':
        image = image_
    else:
        with torch.no_grad():
            if args.half_inference:
                latents = latents.half()
            image = vae.decode(latents).sample.to(torch.float32)
    save_image((image/2+0.5).clamp(0, 1), f'{args.work_dir}/final_image_{image_name}.png')

    if args.generation_mode in ['vsd'] and args.save_phi_model:
        if args.phi_model in ['lora']:
            unet_phi.save_attn_procs(save_directory=f'{args.work_dir}')
        elif args.phi_model in ['unet_simple']:
            unet_phi.save_pretrained(save_directory=f'{args.work_dir}')

#########################################################################################
if __name__ == "__main__":
    main()



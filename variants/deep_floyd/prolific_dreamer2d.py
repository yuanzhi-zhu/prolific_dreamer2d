import os
join = os.path.join
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import io
from tqdm import tqdm
from datetime import datetime
import random
import imageio
from pathlib import Path
from model_utils import (
            get_t_schedule, 
            get_loss_weights, 
            sds_vsd_grad_diffuser, 
            phi_vsd_grad_diffuser, 
            extract_lora_diffusers,
            update_curve,
            get_images,
            )
import shutil
import logging

# from diffusers import StableDiffusionPipeline
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning

from diffusers import UNet2DConditionModel, DiffusionPipeline, DDIMScheduler

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
    parser.add_argument('--log_gif', type=str2bool, default=False, help='Log gif')
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
    parser.add_argument('--init_img_path', type=str, default='', help='init particle from a known image path')
    ### sampling
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps for random sampling')
    parser.add_argument('--t_end', type=int, default=980, help='largest possible timestep for random sampling')
    parser.add_argument('--t_start', type=int, default=20, help='least possible timestep for random sampling')
    parser.add_argument('--multisteps', default=1, type=int, help='multisteps to predict x0')
    parser.add_argument('--t_schedule', default='descend', type=str, help='t_schedule for sampling')
    parser.add_argument('--prompt', default="a photograph of an astronaut riding a horse", type=str, help='prompt')
    parser.add_argument('--height', default=64, type=int, help='height of image')
    parser.add_argument('--width', default=64, type=int, help='width of image')
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
    parser.add_argument('--lora_scale', type=float, default=1.0, help='lora_scale of the unet cross attn')
    parser.add_argument('--use_scheduler', default=False, type=str2bool, help='use_scheduler for lr')
    parser.add_argument('--lr_scheduler_start_factor', type=float, default=1/3, help='Start factor for learning rate scheduler')
    parser.add_argument('--lr_scheduler_iters', type=int, default=300, help='Iterations for learning rate scheduler')
    parser.add_argument('--loss_weight_type', type=str, default='none', help='type of loss weight')
    parser.add_argument('--grad_scale', type=float, default=1., help='grad_scale for loss in vsd')
    args = parser.parse_args()
    # create working directory
    args.run_id = args.run_date + '_' + args.run_time
    args.work_dir = f'{args.work_dir}_{args.run_id}_{args.generation_mode}_cfg_{args.guidance_scale}_bs_{args.batch_size}_num_steps_{args.num_steps}_tschedule_{args.t_schedule}'
    args.work_dir = args.work_dir + f'_{args.phi_model}' if args.generation_mode == 'vsd' else args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)
    assert args.generation_mode in ['t2i', 'sds', 'vsd', 't2i_pipeline', 't2i_stage2']
    assert args.phi_model in ['lora', 'unet_simple']
    if args.init_img_path:
        assert args.batch_size == 1
    # for sds and t2i, use only args.batch_size
    if args.generation_mode in ['t2i', 'sds', 't2i_pipeline', 't2i_stage2']:
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
    
    ### IF stage 2
    if args.generation_mode == 't2i_stage2':
        from diffusers import IFSuperResolutionPipeline
        import gc
        from transformers import T5EncoderModel

        text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder", device_map="auto", variant="fp16"
        )

        # text to image
        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            text_encoder=text_encoder,  # pass the previously instantiated 8bit text encoder
            unet=None,
            device_map="auto",
        )

        # text embeds
        prompt_embeds, negative_embeds = pipe.encode_prompt(args.prompt, device=device)

        # Remove the pipeline so we can re-load the pipeline with the unet
        del text_encoder
        del pipe
        torch.cuda.empty_cache()

        if args.init_img_path:
            # load image
            init_image = io.read_image(args.init_img_path).unsqueeze(0) / 255
            init_image = init_image * 2 - 1   #[-1,1]

        df_IF_stage_2 = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, device_map="auto"
        ).to(device)

        # stage 2
        image = df_IF_stage_2(
                                image=init_image,
                                prompt_embeds=prompt_embeds, \
                                negative_prompt_embeds=negative_embeds, \
                                num_inference_steps=args.num_steps, \
                                guidance_scale=args.guidance_scale, \
                                output_type="pt"
        ).images
        save_image((image/2+0.5).clamp(0, 1), f'{args.work_dir}/final_image_{image_name}.png')
        return

    #######################################################################################
    ### load model
    logger.info(f'load models from path: {args.model_path}')
    df_IF_stage_1 = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    unet = df_IF_stage_1.unet
    if args.generation_mode == 't2i':
        scheduler = df_IF_stage_1.scheduler
    else:
        # for sds and vsd, use DDIMScheduler, only utilize predicted_variance
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler", torch_dtype=dtype)

    if args.half_inference:
        unet = unet.half()
    unet = unet.to(device)
    unet.requires_grad_(False)
    # all variables in same device for scheduler.step()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    if args.generation_mode == 'vsd':
        if args.phi_model == 'lora':
            ### unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            if args.load_phi_model_path:
                unet_phi.load_attn_procs(args.load_phi_model_path)
                unet_phi = unet_phi.to(device)
        elif args.phi_model == 'unet_simple':
            # initialize simple unet, same input/output as (pre-trained) unet
            ### IMPORTANT: need the proper (wide) channel numbers
            unet_phi = UNet2DConditionModel(
                                        sample_size=64,
                                        in_channels=3,
                                        out_channels=3,
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
    elif args.generation_mode == 'sds':
        unet_phi = None
    
    ### get text embedding
    # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
    df_IF_stage_1.text_encoder = df_IF_stage_1.text_encoder.to(device)
    text_embeddings, uncond_embeddings = df_IF_stage_1.encode_prompt(args.prompt, device=device)
    
    text_embeddings_vsd = torch.cat([uncond_embeddings[:args.particle_num_vsd], text_embeddings[:args.particle_num_vsd]])
    text_embeddings_phi = torch.cat([uncond_embeddings[:args.particle_num_phi], text_embeddings[:args.particle_num_phi]])

    ### weight loss
    num_train_timesteps = len(scheduler.betas)
    loss_weights = get_loss_weights(scheduler.betas, args)

    ### scheduler set timesteps
    if args.generation_mode == 't2i':
        scheduler.set_timesteps(args.num_steps)
    else:
        scheduler.set_timesteps(num_train_timesteps)

    ### initialize particles
    if args.use_mlp_particle:
        # use siren network
        from model_utils import Siren
        args.lr = 1e-4
        print(f'for mlp_particle, set lr to {args.lr}')
        out_features = 3
        particles = nn.ModuleList([Siren(2, hidden_features=64, hidden_layers=2, out_features=out_features, device=device) for _ in range(args.batch_size)])
    else:
        if args.init_img_path:
            # load image
            init_image = io.read_image(args.init_img_path).unsqueeze(0) / 255
            init_image = init_image * 2 - 1   #[-1,1]
            particles = init_image.to(device)
        else:
            # gaussian in rgb space
            particles = torch.randn((args.batch_size, 3, args.height, args.width))
    particles = particles.to(device, dtype=dtype)
    # images = images * scheduler.init_noise_sigma
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
    chosen_ts = get_t_schedule(num_train_timesteps, args, loss_weights)
    pbar = tqdm(chosen_ts)
    ### regular sd text to image generation using pipeline
    
    if args.generation_mode == 't2i_pipeline':
        assert not args.log_gif and not args.log_progress
        df_IF_stage_1 = df_IF_stage_1.to(device)

        # text embeds
        prompt_embeds, negative_embeds = text_embeddings, uncond_embeddings

        # stage 1
        image_ = df_IF_stage_1(
                                prompt_embeds=prompt_embeds, \
                                negative_prompt_embeds=negative_embeds, \
                                num_inference_steps=args.num_steps, \
                                guidance_scale=args.guidance_scale, \
                                output_type="pt"
        ).images
        
    ### regular sd text to image generation
    elif args.generation_mode == 't2i':
        if args.phi_model == 'lora' and args.load_phi_model_path:
            ### unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            unet_phi.load_attn_procs(args.load_phi_model_path)
            unet = unet_phi.to(device)
        step = 0
        # get images of all particles
        assert args.use_mlp_particle == False
        images = get_images(particles)
        if args.half_inference:
            images = images.half()
            text_embeddings_vsd = text_embeddings_vsd.half()
        for t in tqdm(scheduler.timesteps):
            # expand the images if we are doing classifier-free guidance to avoid doing two forward passes.
            images_model_input = torch.cat([images] * 2)
            images_model_input = scheduler.scale_model_input(images_model_input, t)
            images_noisy = images
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(images_model_input, t, encoder_hidden_states=text_embeddings_vsd).sample
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            # compute the previous noisy sample x_t -> x_t-1
            # images = scheduler.step(noise_pred, t, images).prev_sample
            images = scheduler.step(noise_pred, t, images)["prev_sample"]
            ######## Evaluation and log metric #########
            if args.log_steps and (step % args.log_steps == 0 or step == (args.num_steps-1)):
                # save current img_tensor
                if args.save_x0:
                    # compute the predicted clean sample x_0
                    pred_images = scheduler.step(noise_pred, t, images_noisy).pred_original_sample.to(dtype).clone().detach()
                if args.half_inference:
                    images = images.half()
                image_ = images.to(torch.float32)
                if args.save_x0:
                    if args.half_inference:
                        pred_images = pred_images.half()
                    image_x0 = pred_images.to(torch.float32)
                    image = torch.cat((image_,image_x0), dim=2)
                else:
                    image = image_
                if args.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
            step += 1
    ### sds text to image generation
    elif args.generation_mode in ['sds', 'vsd']:
        cross_attention_kwargs = {'scale': args.lora_scale} if (args.generation_mode == 'vsd' and args.phi_model == 'lora') else {}
        for step, chosen_t in enumerate(pbar):
            # get latent of all particles
            images = get_images(particles, use_mlp_particle=args.use_mlp_particle)
            t = torch.tensor([chosen_t]).to(device)
            ######## q sample #########
            # random sample particle_num_vsd particles from images
            indices = torch.randperm(images.size(0))
            images_vsd = images[indices[:args.particle_num_vsd]]
            noise = torch.randn_like(images_vsd)
            noisy_images = scheduler.add_noise(images_vsd, noise, t)
            ######## Do the gradient for images!!! #########
            optimizer.zero_grad()
            # loss step
            grad_, noise_pred, noise_pred_phi = sds_vsd_grad_diffuser(unet, noisy_images, noise, text_embeddings_vsd, t, \
                                                    guidance_scale=args.guidance_scale, unet_phi=unet_phi, \
                                                        generation_mode=args.generation_mode, phi_model=args.phi_model, \
                                                            cross_attention_kwargs=cross_attention_kwargs, \
                                                                multisteps=args.multisteps, scheduler=scheduler, \
                                                                    half_inference=args.half_inference, \
                                                                        cfg_phi=args.cfg_phi, grad_scale=args.grad_scale)
            # weighting
            grad_ *= loss_weights[int(t)]
            # ref: https://github.com/threestudio-project/threestudio/blob/5e29759db7762ec86f503f97fe1f71a9153ce5d9/threestudio/models/guidance/stable_diffusion_guidance.py#L427
            # construct loss
            # loss = loss_weights[int(t)] * F.mse_loss(noise_pred, noise, reduction="mean") / args.batch_size
            target = (images_vsd - grad_).detach()
            # d(loss)/d(images) = images - target = images - (images - grad) = grad
            loss = 0.5 * F.mse_loss(images_vsd, target, reduction="mean") / args.batch_size
            loss.backward()
            optimizer.step()
            if args.use_scheduler:
                lr_scheduler.step(loss)

            torch.cuda.empty_cache()
            ######## Do the gradient for unet_phi!!! #########
            if args.generation_mode == 'vsd':
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
                    # random sample particle_num_phi particles from images
                    indices = torch.randperm(images.size(0))
                    images_phi = images[indices[:args.particle_num_phi]]
                    noise_phi = torch.randn_like(images_phi)
                    noisy_images_phi = scheduler.add_noise(images_phi, noise_phi, t_phi)
                    loss_phi = phi_vsd_grad_diffuser(unet_phi, noisy_images_phi.detach(), noise_phi, text_embeddings_phi, t_phi, \
                                                     cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, \
                                                        half_inference=args.half_inference)
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
                if args.save_x0:
                    # compute the predicted clean sample x_0
                    # pred_images = scheduler.step(noise_pred, t, noisy_images).pred_original_sample.to(dtype).clone().detach()
                    pred_images = scheduler.step(noise_pred-noise_pred_phi+noise, t, noisy_images).pred_original_sample.to(dtype).clone().detach()
                    if args.generation_mode == 'vsd':
                        pred_images_phi = scheduler.step(noise_pred_phi, t, noisy_images).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    if args.half_inference:
                        images_vsd = images_vsd.half()
                    image_ = images_vsd.to(torch.float32)
                    if args.save_x0:
                        if args.half_inference:
                            pred_images = pred_images.half()
                        image_x0 = pred_images.to(torch.float32)
                        if args.generation_mode == 'vsd':
                            if args.half_inference:
                                pred_images_phi = pred_images_phi.half()
                            image_x0_phi = pred_images_phi.to(torch.float32)
                            image = torch.cat((image_,image_x0,image_x0_phi), dim=2)
                        else:
                            image = torch.cat((image_,image_x0), dim=2)
                    else:
                        image = image_
                if args.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
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

    if args.log_gif:
        # make gif
        images = sorted(Path(args.work_dir).glob(f"*{image_name}*.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(f'{args.work_dir}/{image_name}.gif', images, duration=0.3)
    if args.log_progress and args.batch_size == 1:
        concatenated_images = torch.cat(image_progress, dim=0)
        save_image(concatenated_images, f'{args.work_dir}/{image_name}_prgressive.png')
    # save final image
    if 't2i' in args.generation_mode:
        image = image_
    else:
        image = get_images(particles, use_mlp_particle=args.use_mlp_particle)
    save_image((image/2+0.5).clamp(0, 1), f'{args.work_dir}/final_image_{image_name}.png')

    if args.generation_mode in ['vsd'] and args.save_phi_model:
        if args.phi_model in ['lora']:
            unet_phi.save_attn_procs(save_directory=f'{args.work_dir}')
        elif args.phi_model in ['unet_simple']:
            unet_phi.save_pretrained(save_directory=f'{args.work_dir}')

#########################################################################################
if __name__ == "__main__":
    main()



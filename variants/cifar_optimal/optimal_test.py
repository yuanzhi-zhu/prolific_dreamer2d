# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

import os
import imageio
from pathlib import Path
import argparse
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import torchvision
from torchvision import io
from torchvision.utils import save_image
import random
from torch.utils.data import DataLoader

### optimal score for VE trajectory
def optimal_score(noisy_latents, particles, sigma):
    ### predict optimal output based on the GT data points: -\nabla U ###
    sigma_2 = sigma ** 2
    def gauss_norm(z, xi): # much faster when pixels are *uncorelated*
        gauss =  torch.exp( -(z - xi)**2 / (2 * sigma_2) ) / ( sigma * torch.sqrt(torch.tensor(2*torch.pi)) )
        log_gauss = torch.log(gauss)
        return log_gauss.sum(dim=list(range(1, len(log_gauss.shape))))
    log_gauss_pdf_list = []
    for i in range(particles.shape[0]):
        log_gauss_pdf_list.append(gauss_norm(noisy_latents, particles[i]))
    log_gauss_pdfs = torch.stack(log_gauss_pdf_list, dim=0).T
    post_softmax = torch.nn.functional.softmax(log_gauss_pdfs, dim=1)
    post_softmax = torch.nan_to_num(post_softmax, nan=0.0)
    weighted_softmax = torch.einsum('bc,cijk->bijk', post_softmax, particles)
    return 1 / sigma_2 * (weighted_softmax - noisy_latents)

### optimal score for VP trajectory
def optimal_noise(noisy_latents, particles, alpha):
    sigma = torch.sqrt(1 - alpha ** 2)
    sigma_2 = 1 - alpha ** 2
    def gauss_norm(z, xi):
        # much faster when pixels are *uncorelated*
        gauss =  torch.exp( -(z - alpha * xi)**2 / (2 * sigma_2) ) / ( sigma * torch.sqrt(torch.tensor(2*torch.pi)) )
        log_gauss = torch.log(gauss)
        log_gauss_pdf = log_gauss.sum(dim=list(range(1, len(log_gauss.shape))))
        return log_gauss_pdf
    log_gauss_pdf_list = []
    for i in range(particles.shape[0]):
        log_gauss_pdf_list.append(gauss_norm(noisy_latents, particles[i]))
    log_gauss_pdfs = torch.stack(log_gauss_pdf_list, dim=0).T
    # max_values, indices = torch.max(log_gauss_pdfs, dim=1)
    # # eliminate self influence
    # for i in range(indices.shape[0]):
    #     log_gauss_pdfs[i, indices[i]] = -torch.inf
    post_softmax = torch.nn.functional.softmax(log_gauss_pdfs, dim=1)
    post_softmax = torch.nan_to_num(post_softmax, nan=0.0)
    weighted_softmax = alpha * torch.einsum('bc,cijk->bijk', post_softmax, particles)
    optimal_score = 1 / sigma_2 * (weighted_softmax - noisy_latents)
    return - sigma * optimal_score


def sigma_schedule(args):
    """
    Generate a list of sigma values with a specified schedule.
        list: A list of sigma values with length n_iters.
    """
    # Define the schedule function based on the selected type
    if args.sigma_schedule == 'linear':
        def schedule_func(t):
            return args.sigma_min + (args.sigma_max - args.sigma_min) * t / (args.num_sigmas - 1)
    elif args.sigma_schedule == 'karras':
        """Implementation of the karras noise schedule"""
        def schedule_func(t):
            rho = 7
            rho_inv = 1.0 / rho
            sigma = args.sigma_min**rho_inv + t / max(args.num_sigmas - 1, 1) * (
                args.sigma_max**rho_inv - args.sigma_min**rho_inv
            )
            return sigma**rho
    else:
        raise ValueError("Invalid schedule_type. Available options are 'linear', 'exponential', or 'cosine'.")
    # Generate the sigma values based on the schedule function
    sigma_values = [schedule_func(t) for t in range(args.num_sigmas)]
    return sigma_values


def get_t_schedule(args):
    # Create a list of time steps from 0 to num_train_timesteps
    ts = list(range(args.num_sigmas))
    # set ts to U[0.02,0.98] as least
    assert (args.t_start >= 20) and (args.t_end <= 980)
    ts = ts[args.t_start:args.t_end]

    # If the scheduling strategy is 'random', choose n_iters random time steps without replacement
    if args.t_schedule == 'random':
        chosen_ts = np.random.choice(ts, args.n_iters, replace=True)

    # If the scheduling strategy is 'random_down', first exclude the first 30 and last 10 time steps
    # then choose a random time step from an interval that shrinks as step increases
    elif 'random_down' in args.t_schedule:
        interval_ratio = int(args.t_schedule[11:]) if len(args.t_schedule[11:]) > 0 else 5
        interval_ratio *= 0.1 
        chosen_ts = [np.random.choice(
                        ts[max(0,int((args.n_iters-step-interval_ratio*args.n_iters)/args.n_iters*len(ts))):\
                           min(len(ts),int((args.n_iters-step+interval_ratio*args.n_iters)/args.n_iters*len(ts)))], 
                     1, replace=True).astype(int)[0] \
                     for step in range(args.n_iters)]

    # If the scheduling strategy is 'fixed', parse the fixed time step from the string and repeat it n_iters times
    elif 'fixed' in args.t_schedule:
        fixed_t = int(args.t_schedule[5:])
        chosen_ts = [fixed_t for _ in range(args.n_iters)]

    # If the scheduling strategy is 'descend', parse the start time step from the string (or default to 1000)
    # then create a list of descending time steps from the start to 0, with length n_iters
    elif 'descend' in args.t_schedule:
        if 'quad' in args.t_schedule:   # no significant improvement
            descend_from = int(args.t_schedule[12:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.square(np.linspace(descend_from**0.5, 1, args.n_iters))
            chosen_ts = chosen_ts.astype(int).tolist()
        else:
            descend_from = int(args.t_schedule[7:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.linspace(descend_from-1, 1, args.n_iters, endpoint=True)
            chosen_ts = chosen_ts.astype(int).tolist()

    # If the scheduling strategy is 't_stages', the total number of time steps are divided into several stages.
    # In each stage, a decreasing portion of the total time steps is considered for selection.
    # For each stage, time steps are randomly selected with replacement from the respective portion.
    # The final list of chosen time steps is a concatenation of the time steps selected in all stages.
    # Note: The total number of time steps should be evenly divisible by the number of stages.
    elif 't_stages' in args.t_schedule:
        # Parse the number of stages from the scheduling strategy string
        num_stages = int(args.t_schedule[8:]) if len(args.t_schedule[8:]) > 0 else 2
        chosen_ts = []
        for i in range(num_stages):
            # Define the portion of ts to be considered in this stage
            portion = ts[:int((num_stages-i)*len(ts)//num_stages)]
            selected_ts = np.random.choice(portion, args.n_iters//num_stages, replace=True).tolist()
            chosen_ts += selected_ts
    
    else:
        raise ValueError(f"Unknown scheduling strategy: {args.t_schedule}")

    # Return the list of chosen time steps
    return chosen_ts

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', default='consistency_models', type=str, help='model folder name')
    parser.add_argument('--seed', default=42, type=int, help='global seed')
    parser.add_argument('--noisy_input', default=1, type=int, help='use noisy_input to calc score') 
    parser.add_argument('--train_dataset', default=1, type=int, help='use train_dataset') 
    parser.add_argument('--desired_class', default='cat', type=str, help='desired_class') 
    parser.add_argument('--n_samples', default=64, type=int, help='number of particles') 
    parser.add_argument('--n_iters', default=1000, type=int, help='number of iters') 
    parser.add_argument('--inner_iter', default=1, type=int, help='inner_iter')
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--steps_per_frame', default=20, type=int, help='steps_per_frame')
    parser.add_argument('--num_sigmas', type=int, default=1000, help='Number of steps for random sampling')
    parser.add_argument('--lr', default=0.01, type=float, help='lr')
    parser.add_argument('--init_img_path', type=str, default='', help='init particle from a known image path')
    parser.add_argument('--t_end', type=int, default=980, help='largest possible timestep for random sampling')
    parser.add_argument('--t_start', type=int, default=20, help='least possible timestep for random sampling')
    parser.add_argument('--t_schedule', default='random', type=str, help='t_schedule')
    parser.add_argument('--sigma_max', default=80, type=float, help='sigma_max')
    parser.add_argument('--sigma_min', default=0.01, type=float, help='sigma_min')
    parser.add_argument('--sigma_schedule', default='karras', type=str, help='sigma_schedule')
    args = parser.parse_args()
    # set random seed everywhere
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    return args

def main():
    ### working path & setup logger
    args = parse_args_and_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # gt points & particle initialization
    ###################### load datasets ######################
    dataset = torchvision.datasets.CIFAR10(root='datasets/cifar', download=True, train=args.train_dataset,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.ToTensor(),
                                                # torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                        ),)
    # CIFAR10 class labels
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Filter the dataset to only keep 'cat' images
    if args.desired_class != 'all':
        class_idx = classes.index(args.desired_class)
        dataset = [(img, label) for img, label in dataset if label == class_idx]
    # Use DataLoader to efficiently load and process images in batches
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # loop dataloader
    all_images = []
    print('loading dataset')
    # Iterate through the data_loader to access batches of images as tensors
    for batch_images, batch_labels in tqdm.tqdm(data_loader):
        # Perform any additional processing if needed
        # For example, you can directly use batch_images in your model for inference or training
        all_images.append(batch_images)
    batch_tensor = torch.cat(all_images, dim=0)
    gt_images = batch_tensor.to(device)
    print(f'length of gt_iamges: {gt_images.shape[0]}')

    ###################### initial particles ######################
    def init_particles():
        if args.init_img_path:
            # load image
            init_image = io.read_image(args.init_img_path).unsqueeze(0) / 255
            particles = init_image * 2 - 1   #[-1,1]
            particles = particles.to(device)
        else:
            particles = torch.randn_like(gt_images[:args.n_samples])
        return particles

    ### scheduler
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="scheduler")
    scheduler.set_timesteps(1000)
    # scheduler.set_timesteps(args.n_iters)


    ###################### main program ######################
    # mode = 'VE': Langevin dynamics with VE scheme
    # mode = 'VP': Langevin dynamics with VP scheme
    # mode = 'SDS': Score Distillation Sampling
    def run_model(particles, mode='VE', fix_t=100, noisy_input=args.noisy_input):
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        work_dir = f'output/{args.expr}_{run_id}_lr{args.lr}_inner_iter{args.inner_iter}_noisy_input{noisy_input}'
        work_dir = f'{work_dir}_{mode}'
        os.makedirs(work_dir, exist_ok=True)
        sigmas = sigma_schedule(args)
        choosen_ts = get_t_schedule(args)
        pbar = tqdm.tqdm(choosen_ts)
        counter = 0
        
        if mode == 'SDS':
            particles.requires_grad = True
            particles_to_optimize = [particles]
            optimizer = torch.optim.Adam(particles_to_optimize, lr=args.lr)
        else:
            particles.requires_grad = False

        for t in pbar:
            # sigma = 0.02 if t < 50 else 0.01 # set current noise level: annealing down
            # sigma = sigmas[t]
            for _ in range(args.inner_iter): # inner loop
                if mode == 'VE':
                    # one step Langevin dynamics with VE scheme
                    sigma = sigmas[t] if fix_t == 0 else sigmas[fix_t]
                    if noisy_input > 0:
                        noise = torch.randn_like(particles)
                        noisy_particles = (particles) + sigma * torch.randn_like(particles)
                    else:
                        noisy_particles = particles.clone()
                    ## Score-Difference Flow
                    # delta_z = optimal_score(noisy_particles, gt_images, sigma) - \
                    #             optimal_score(noisy_particles, particles, sigma)
                    delta_z = optimal_score(noisy_particles, gt_images, sigma)
                    particles += args.lr * (delta_z * sigma ** 2) #+ np.sqrt(2*args.lr) * sigma * torch.randn_like(particles)
                    # clean_x0 = (noisy_particles + delta_z * sigma ** 2)

                elif mode == 'VP':
                    # one step Langevin dynamics with VP scheme
                    alpha = torch.sqrt(scheduler.alphas_cumprod[t]) if fix_t == 0 else torch.sqrt(scheduler.alphas_cumprod[fix_t])
                    if noisy_input > 0:
                        noise = torch.randn_like(particles)
                        noisy_particles = alpha * (particles) + torch.sqrt(1 - alpha ** 2) * noise
                    else:
                        noisy_particles = particles.clone()
                    delta_z = optimal_noise(noisy_particles, gt_images, alpha)
                    delta_z = - delta_z / torch.sqrt(1 - alpha ** 2)
                    particles += args.lr * (delta_z * torch.sqrt(1 - alpha ** 2) ** 2) #+ np.sqrt(2*args.lr) * sigma * torch.randn_like(particles)

                elif mode == 'SDS':
                    # one step SDS
                    alpha = torch.sqrt(scheduler.alphas_cumprod[t]) if fix_t == 0 else torch.sqrt(scheduler.alphas_cumprod[fix_t])
                    t = fix_t if fix_t > 0 else t
                    optimizer.zero_grad()
                    if noisy_input > 0:
                        noise = torch.randn_like(particles)
                        noisy_particles = alpha * (particles) + torch.sqrt(1 - alpha ** 2) * noise
                        grad_ = optimal_noise(noisy_particles.clone().detach(), gt_images, alpha)
                        grad_ = grad_ - noise
                    else:
                        grad_ = optimal_noise(particles.clone().detach(), gt_images, alpha)
                    
                    target = (particles - grad_).detach()
                    # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
                    loss = 0.5 * F.mse_loss(particles, target, reduction="mean") / particles.shape[0]
                    loss.backward()
                    optimizer.step()
                    # particles -= args.lr * (delta_z) #* sigma ** 2 #+ np.sqrt(2*args.lr) * sigma * torch.randn_like(particles)
            
            if counter % args.steps_per_frame == 0 or (counter == args.n_iters-1):
                save_image(particles.clone()/2+0.5, f"{work_dir}/{counter:06d}_t{t:03d}_img.png")
                # save_image(noisy_particles.clone()/2+0.5, f"{work_dir}/{counter:06d}_t{t:03d}_img.png")
            counter += 1

        # make gif
        images = sorted(Path(work_dir).glob("*_img.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(f'{work_dir}/final.gif', images, duration=0.05)


    ###################### run!!! ######################
    particles = init_particles()
    run_model(particles, mode='SDS', fix_t=0, noisy_input=0)
    particles = init_particles()
    run_model(particles, mode='SDS', fix_t=0, noisy_input=1)
    particles = init_particles()
    run_model(particles, mode='VP', fix_t=100, noisy_input=0)
    particles = init_particles()
    run_model(particles, mode='VP', fix_t=100, noisy_input=1)
    particles = init_particles()
    run_model(particles, mode='VE', fix_t=100, noisy_input=0)
    particles = init_particles()
    run_model(particles, mode='VE', fix_t=100, noisy_input=1)
    
    # ###################### test optimal noise --> no problem ######################
    # # particles = torch.randn_like(gt_images[:1])
    # particles = gt_images[:1]
    # alpha = torch.sqrt(scheduler.alphas_cumprod[500])
    # noisy_particles = alpha * (particles) + torch.sqrt(1 - alpha ** 2) * torch.randn_like(particles)
    # noise_pred = optimal_noise(noisy_particles, gt_images, alpha)
    # test = (noisy_particles - torch.sqrt(1 - alpha ** 2) * noise_pred) / alpha
    # save_image(test, f"{work_dir}/test_particles.png")

    # ###################### regular ddim image generation ######################
    # step = 0
    # latents = particles
    # for i in tqdm.tqdm(range(len(scheduler.timesteps))):
    #     t = scheduler.timesteps[i]
    #     latent_noisy = latents
    #     # predict the noise residual
    #     with torch.no_grad():
    #         alpha = torch.sqrt(scheduler.alphas_cumprod[t-1])
    #         noise_pred = optimal_noise(latents, gt_images, alpha)
    #         # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_vsd).sample
        
    #     ## compute the previous noisy sample x_t -> x_t-1
    #     latents = scheduler.step(noise_pred, t-1, latents).prev_sample
    #     latents_x0 = scheduler.step(noise_pred, t-1, latents).pred_original_sample

    #     # ## without schuduler
    #     # latents_x0 = (latents - torch.sqrt(1 - alpha ** 2) * noise_pred) / alpha
    #     # if t == 1:
    #     #     latents = latents_x0
    #     # else:
    #     #     tm1 = scheduler.timesteps[i+1]
    #     #     alpha_tm1 = torch.sqrt(scheduler.alphas_cumprod[tm1-1])
    #     #     # ddim
    #     #     latents = alpha_tm1 * latents_x0 + torch.sqrt(1 - alpha_tm1 ** 2) * noise_pred
        
    #     ######## Evaluation and log metric #########
    #     if step % args.steps_per_frame == 0 or (step == args.n_iters-1):
    #         save_image(latents.clone(), f"{work_dir}/{step:06d}_t{t:03d}_particles.png")
    #         save_image(latents_x0.clone(), f"{work_dir}/{step:06d}_t{t:03d}_particles_x0.png")
    #     step += 1
    
if __name__ == "__main__":
    main()
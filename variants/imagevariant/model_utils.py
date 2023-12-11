import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        # return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)
        # return the mse loss (grad_scale * (noise_pred - noise_pred_phi.detach()))**2: not a good indicator
        return torch.mean(gt_grad**2).to(device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def get_t_schedule(num_train_timesteps, args, loss_weight=None):
    # Create a list of time steps from 0 to num_train_timesteps
    ts = list(range(num_train_timesteps))
    # set ts to U[0.02,0.98] as least
    assert (args.t_start >= 20) and (args.t_end <= 980)
    ts = ts[args.t_start:args.t_end]

    # If the scheduling strategy is 'random', choose args.num_steps random time steps without replacement
    if args.t_schedule == 'random':
        chosen_ts = np.random.choice(ts, args.num_steps, replace=True)

    # If the scheduling strategy is 'random_down', first exclude the first 30 and last 10 time steps
    # then choose a random time step from an interval that shrinks as step increases
    elif 'random_down' in args.t_schedule:
        interval_ratio = int(args.t_schedule[11:]) if len(args.t_schedule[11:]) > 0 else 5
        interval_ratio *= 0.1 
        chosen_ts = [np.random.choice(
                        ts[max(0,int((args.num_steps-step-interval_ratio*args.num_steps)/args.num_steps*len(ts))):\
                           min(len(ts),int((args.num_steps-step+interval_ratio*args.num_steps)/args.num_steps*len(ts)))], 
                     1, replace=True).astype(int)[0] \
                     for step in range(args.num_steps)]

    # If the scheduling strategy is 'fixed', parse the fixed time step from the string and repeat it args.num_steps times
    elif 'fixed' in args.t_schedule:
        fixed_t = int(args.t_schedule[5:])
        chosen_ts = [fixed_t for _ in range(args.num_steps)]

    # If the scheduling strategy is 'descend', parse the start time step from the string (or default to 1000)
    # then create a list of descending time steps from the start to 0, with length args.num_steps
    elif 'descend' in args.t_schedule:
        if 'quad' in args.t_schedule:   # no significant improvement
            descend_from = int(args.t_schedule[12:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.square(np.linspace(descend_from**0.5, 1, args.num_steps))
            chosen_ts = chosen_ts.astype(int).tolist()
        else:
            descend_from = int(args.t_schedule[7:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.linspace(descend_from-1, 1, args.num_steps, endpoint=True)
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
            selected_ts = np.random.choice(portion, args.num_steps//num_stages, replace=True).tolist()
            chosen_ts += selected_ts
    
    elif 'dreamtime' in args.t_schedule:
        # time schedule in dreamtime https://arxiv.org/abs//2306.12422
        assert 'dreamtime' in args.loss_weight_type
        loss_weight_sum = np.sum(loss_weight)
        p = [wt / loss_weight_sum for wt in loss_weight]
        N = args.num_steps
        def t_i(t, i, p):
            t = int(max(0, min(len(p)-1, t)))
            return abs(sum(p[t:]) - i/N)
        chosen_ts = []
        for i in range(N):
            # Initial guess for t
            t0 = 999
            selected_t = minimize(t_i, t0, args=(i, p), method='Nelder-Mead')
            selected_t = max(0, int(selected_t.x))
            chosen_ts.append(selected_t)
    else:
        raise ValueError(f"Unknown scheduling strategy: {args.t_schedule}")

    # Return the list of chosen time steps
    return chosen_ts


def loss_weights(betas, args):
    num_train_timesteps = len(betas)
    betas = torch.tensor(betas) if not torch.is_tensor(betas) else betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image
    sigma_ks = []
    SNRs = []
    rhos = []
    m1 = 800
    m2 = 500
    s1 = 300
    s2 = 100
    for i in range(num_train_timesteps):
        sigma_ks.append(reduced_alpha_cumprod[i])
        SNRs.append(1 / reduced_alpha_cumprod[i])
        if args.loss_weight_type == 'rhos':
            rhos.append(1. * (args.sigma_y**2)/(sigma_ks[i]**2))
    def loss_weight(t):
        if args.loss_weight_type == None or args.loss_weight_type == 'none':
            return 1
        elif 'SNR' in args.loss_weight_type:
            if args.loss_weight_type == 'SNR':
                return 1 / SNRs[t]
            elif args.loss_weight_type == 'SNR_sqrt':
                return np.sqrt(1 / SNRs[t])
            elif args.loss_weight_type == 'SNR_square':
                return (1 / SNRs[t])**2
            elif args.loss_weight_type == 'SNR_log1p':
                return np.log(1 + 1 / SNRs[t])
        elif args.loss_weight_type == 'rhos':
            return 1 / rhos[t]
        elif 'alpha' in args.loss_weight_type:
            if args.loss_weight_type == 'sqrt_alphas_cumprod':
                return sqrt_alphas_cumprod[t]
            elif args.loss_weight_type == '1m_alphas_cumprod':
                return sqrt_1m_alphas_cumprod[t]**2
            elif args.loss_weight_type == 'alphas_cumprod':
                return alphas_cumprod[t]
            elif args.loss_weight_type == 'sqrt_alphas_1m_alphas_cumprod':
                return sqrt_alphas_cumprod[t] * sqrt_1m_alphas_cumprod[t]
        elif 'dreamtime' in args.loss_weight_type:
            if t > m1:
                return np.exp(-(t - m1)**2 / (2 * s1**2))
            elif t >= m2:
                return 1
            else:
                return np.exp(-(t - m2)**2 / (2 * s2**2))
        else:
            raise NotImplementedError
    weights = []
    for i in range(num_train_timesteps):
        weights.append(loss_weight(i))
    return weights


def predict_noise0_diffuser_instruct(unet, noisy_latents, text_embeddings, t, image_latents=None, guidance_scale=7.5, img_guidance_scale=1.5, scheduler=None, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 3)
    scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

    text_embeddings = torch.cat([text_embeddings[:1], text_embeddings])
    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(scaled_latent_model_input[batch_size:], t, encoder_hidden_states=text_embeddings[batch_size:], return_dict=False).sample
    else:
        noise_pred = unet(scaled_latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
        # predict the noise residual
        noise_pred_uncond, noise_pred_image, noise_pred_text = noise_pred.chunk(3)
        # perform guidance
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + img_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
    noise_pred = noise_pred.float()
    return noise_pred

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if lora_v:
        # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
    else:
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred


def predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, steps=1, eta=0, half_inference=False):
    latents = noisy_latents
    # get sub-sequence with length step_size
    t_start = t.item()
    # Ensure that t and steps are within the valid range
    if not (0 < t_start <= 1000):
        raise ValueError(f"t must be between 0 and 1000, get {t_start}")
    if t_start > steps:
        # Calculate the step size
        step_size = t_start // steps
        # Generate a list of indices
        indices = [int((steps - i) * step_size) for i in range(steps)]
        if indices[0] != t_start:
            # indices.insert(0, t_start)    # add start point
            indices[0] = t_start    # replace start point
    else:
        indices = [int((t_start - i)) for i in range(t_start)]
    if indices[-1] != 0:
        indices.append(0)
    # run multistep ddim sampling
    for i in range(len(indices)):
        t = torch.tensor([indices[i]] * t.shape[0], device=t.device)
        noise_pred = predict_noise0_diffuser(unet, latents, text_embeddings, t, guidance_scale=guidance_scale, \
                                             cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)
        if i == 0:
            noise_pred_1step = noise_pred.clone().detach()
        pred_latents = scheduler.step(noise_pred, t, latents).pred_original_sample
        if indices[i+1] == 0:
            ### use pred_latents and latents calculate equivalent noise_pred
            alpha_bar_t_start = scheduler.alphas_cumprod[indices[0]].clone().detach()
            return (noisy_latents - torch.sqrt(alpha_bar_t_start)*pred_latents) / (torch.sqrt(1 - alpha_bar_t_start)), noise_pred_1step
        alpha_bar = scheduler.alphas_cumprod[indices[i]].clone().detach()
        alpha_bar_prev = scheduler.alphas_cumprod[indices[i+1]].clone().detach()
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(latents)
        mean_pred = (
            pred_latents * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(latents.shape) - 1)))
        )  # no noise when t == 0
        latents = mean_pred + nonzero_mask * sigma * noise
    # return out["pred_xstart"]


def sds_vsd_grad_diffuser(unet, latents, noise, text_embeddings, t, unet_phi=None, guidance_scale=7.5, \
                        grad_scale=1, cfg_phi=1., generation_mode='sds', phi_model='lora', \
                            cross_attention_kwargs={}, multisteps=1, scheduler=None, lora_v=False, \
                                half_inference = False, text_embeddings_phi=None, grad_scale_phi=1., \
                                    image_latents=None):
    # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114
    unet_cross_attention_kwargs = {'scale': 0} if (generation_mode == 'vsd' and phi_model == 'lora' and not lora_v) else {}
    text_embeddings_phi = text_embeddings if text_embeddings_phi == None else text_embeddings_phi
    with torch.no_grad():
        # predict the noise residual with unet
        # set cross_attention_kwargs={'scale': 0} to use the pre-trained model
        if multisteps >= 1:
            noise_pred, noise_pred_1step = predict_noise0_diffuser_multistep(unet, latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs, scheduler=scheduler, steps=multisteps, eta=0., half_inference=half_inference)
        else:
            noise_pred = predict_noise0_diffuser(unet, latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)

    if generation_mode == 'sds':
        # SDS
        noise = noise - 1 * torch.randn_like(noise)
        grad = grad_scale * (noise_pred - noise)
        # grad = grad_scale * (noise_pred)
        noise_pred_phi = noise
    elif generation_mode == 'vsd':
        with torch.no_grad():
            if phi_model == 'instruct_p2p':
                noise_pred_phi = predict_noise0_diffuser_instruct(unet_phi, latents, text_embeddings_phi, t, image_latents=image_latents, guidance_scale=cfg_phi, img_guidance_scale=1.5, scheduler=scheduler, half_inference=half_inference)
            else:
                noise_pred_phi = predict_noise0_diffuser(unet_phi, latents, text_embeddings_phi, t, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, lora_v=lora_v, half_inference=half_inference)
        # VSD
        # alphas_cumprod = scheduler.alphas_cumprod.to(
        #     device=latents.device, dtype=latents.dtype
        # )
        # alpha_t = alphas_cumprod[t] ** 0.5
        # sigma_t = (1 - alphas_cumprod[t]) ** 0.5
        # noise_pred_phi = sigma_t * noise + alpha_t * noise_pred_phi
        # noise_pred_phi = alpha_t * noise + sigma_t * noise_pred_phi
        # noise_pred_phi = noise
        # grad = grad_scale * (noise_pred - noise_pred_phi.detach())
        # grad = grad_scale * ( (noise_pred - noise) + grad_scale_phi * (noise - noise_pred_phi) )
        grad = grad_scale * ( (noise_pred) + grad_scale_phi * (- noise_pred_phi) )
        # grad = grad_scale * ( (noise_pred_1step) + grad_scale_phi * (- noise_pred) )

    grad = torch.nan_to_num(grad)
    # since we omitted an item in grad, we need to use the custom function to specify the gradient
    loss = SpecifyGradient.apply(latents, grad)

    return loss, noise_pred.detach().clone(), noise_pred_phi.detach().clone()

def phi_vsd_grad_diffuser(unet_phi, latents, noise, text_embeddings, t, cfg_phi=1., grad_scale=1, cross_attention_kwargs={}, scheduler=None, lora_v=False):
    loss_fn = nn.MSELoss()
    # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114
    # predict the noise residual with unet
    clean_latents = scheduler.step(noise, t, latents).pred_original_sample
    noise_pred = predict_noise0_diffuser(unet_phi, latents, text_embeddings, t, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler)
    if lora_v:
        target = scheduler.get_velocity(clean_latents.detach(), noise, t)
    else:
        target = noise
    loss = loss_fn(noise_pred, target)
    loss *= grad_scale

    return loss


def extract_lora_diffusers(unet, device):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        ).to(device)
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # self.unet.requires_grad_(True)
    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)
    # self.params_to_optimize = unet_lora_layers.parameters()
    ### end lora
    return unet, unet_lora_layers


def update_curve(values, label, x_label, y_label, model_path, run_id, log_steps=None):
    fig, ax = plt.subplots()
    if log_steps:
        ax.plot(log_steps, values, label=label)
    else:
        ax.plot(values, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(f'{model_path}/{label}_curve_{run_id}.png', dpi=600)
    plt.close()


### siren from https://github.com/vsitzmann/siren/
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, device, \
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        self.device = device
        self.out_features = out_features

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output #, coords
    
    def generate_image(self, img_size=64):
        # Generate an input grid coordinates in a range of -1 to 1.
        grid = torch.Tensor([[[2*(x / (img_size - 1)) - 1, 2*(y / (img_size - 1)) - 1] for y in range(img_size)] for x in range(img_size)])
        grid = grid.view(-1, 2)  # Reshape to (img_size*img_size, 2)
        grid = grid.to(self.device)
        rgb_values = self.forward(grid)
        rgb_values = torch.tanh(rgb_values)     # [-1, 1]
        # Reshape to an image
        rgb_values = rgb_values.view(1, img_size, img_size, self.out_features)
        image = rgb_values.permute(0, 3, 1, 2)
        return image

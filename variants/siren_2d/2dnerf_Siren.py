
import os
from PIL import Image
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt


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
                # final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                #                               np.sqrt(6 / hidden_features) / hidden_omega_0)
                final_linear.weight.normal_(0, 1 / hidden_features)
            self.net.append(final_linear)
            # self.net.append(nn.Tanh())
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
        # rgb_values = torch.tanh(rgb_values)     # [-1, 1]
        # Reshape to an image
        rgb_values = rgb_values.view(1, img_size, img_size, self.out_features)
        image = rgb_values.permute(0, 3, 1, 2)
        return image



def get_latents(particles, rgb_as_latents=False, use_mlp_particle=True, output_size=512):
    ### get latents from particles
    if use_mlp_particle:
        images = []
        output_size = output_size // 8 if rgb_as_latents else output_size
        # Loop over all MLPs and generate an image for each
        for particle_mlp in particles:
            image = particle_mlp.generate_image(output_size)
            images.append(image)
        # Stack all images together
        latents = torch.cat(images, dim=0)
    return latents

print('loading image...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = Image.open('./mmexport1687868523952.jpg')

convert_tensor = transforms.ToTensor()

img = convert_tensor(img)

img = img.unsqueeze(0)

img = img*2-1

img = img.to(device)

output_size = img.shape[-1]


from tqdm import tqdm
def run_expr(hidden_layers=2, lr = 0.002, iters=500, work_dir=''):
    # particles = nn.ModuleList([MLP(device, rgb_as_latents=False) for _ in range(1)])
    particles = nn.ModuleList([Siren(2, hidden_features=128, hidden_layers=hidden_layers, out_features=3, device=device) for _ in range(1)])
    particles = particles.to(device)
    particles_to_optimize = [param for mlp in particles for param in mlp.parameters() if param.requires_grad]
    total_parameters = sum(p.numel() for p in particles_to_optimize if p.requires_grad)
    print(f'Total number of trainable parameters in particles: {total_parameters}; number of particles: {1}')
    optimizer = torch.optim.Adam(particles_to_optimize, lr=lr)
    train_loss_values = []
    pbar = tqdm(range(iters))
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    for step in pbar:
        optimizer.zero_grad()
        out = get_latents(particles, False, output_size=output_size)
        loss = loss_fn(img, out)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Loss: {loss.item():.6f}')
        train_loss_values.append(loss.item())
        if step % 50 == 0 or step == iters-1:
            save_image((out/2+0.5).clamp(0, 1),f'{work_dir}/step{step}.png')
    return train_loss_values

for hd in [2]:
    train_loss_valuesss = []
    # lrs from 0.003 to 0.0045, with increment 0.0001
    # lrs = [0.003 + 0.0001*i for i in range(15)]
    work_dir = f'hl{hd}'
    lrs = [0.004, 0.008]
    for lr in lrs:
        work_dir_son = f'{work_dir}/lr{lr}'
        train_loss_values = run_expr(hidden_layers=hd, lr=lr, work_dir=work_dir_son)
        train_loss_valuesss.append(train_loss_values)
    # plot all curves in one figure
    fig, ax = plt.subplots()
    for i, train_loss_values in enumerate(train_loss_valuesss):
        ax.plot(train_loss_values, label=f'lr={lrs[i]}')
    ax.set_xlabel('steps')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f'{work_dir}/train_loss_hl{hd}.png', dpi=600)
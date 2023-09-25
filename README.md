# Unofficial implementation of 2D ProlificDreamer

This is a third-party implementation of the 2D demos in the paper: [ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation](https://arxiv.org/abs/2305.16213).

This code does *NOT* serve as a faithful re-implementation, and please feel free to raise issues for potential bugs.

Feel free to try the colab notebook https://github.com/yuanzhi-zhu/prolific_dreamer2d/blob/main/prolific_dreamer_2D.ipynb

<p align="center">
  <img src="figs/illustration.png" width="900px"/><br/>
  <em>Progress of generation with VSD sampling (1 particle, CFG=7.5)</em>
</p>

<p align="center">
  <img src="figs/16particles.png" width="900px"/><br/>
  <em>Final results of 16 particles (VSD)</em>
</p>

<p align="center">
  <img src="figs/comparison.png" width="900px"/><br/>
  <em>Comparison between SDS, VSD and Text2Image with different CFG</em>
</p>

## TODO List
- [ ] SGD optimizer does not work for sds
- [ ] make mlp particle work
- [ ] make DeepFloyd IF (SDS & VSD) work

## Commands to Reproduce Results
#### VSD command line
```python
python prolific_dreamer2d.py \
        --num_steps 500 --log_steps 50 \
        --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'runwayml/stable-diffusion-v1-5' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true --save_phi_model true \
```

#### VSD command line multiple particles
```python
python prolific_dreamer2d.py \
        --num_steps 1500 --log_steps 50 \
        --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 16 --guidance_scale 7.5 \
        --particle_num_vsd 2 --particle_num_phi 2 \
        --log_progress false --save_x0 false --save_phi_model true
```

#### SDS command line
```python
python prolific_dreamer2d.py \
        --num_steps 500 --log_steps 50 \
        --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'runwayml/stable-diffusion-v1-5' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
```

## Short Intro to Important Parameters
**generation_mode**: 'sds' or 'vsd' or 't2i' (just text-to-image sampling)

**guidance_scale**: CFG scale

**phi_model**: a lora model or simple unet model to track the (particles) distribution

**num_steps**: overall sampling steps

**use_t_phi**: use different t to train phi model

**loss_weight**: which weight to use for SDS/VSD loss, see https://github.com/yuanzhi-zhu/prolific_dreamer2d/blob/main/model_utils.py#L94

**t_schedule**: generate a sequence of timesteps, see https://github.com/yuanzhi-zhu/prolific_dreamer2d/blob/main/model_utils.py#L17; by default we use 'random', to use the 2 stage time schedule $U[0.02,0.98] \rightarrow U[0.5,0.98]$ as in the paper, we can use 't_stages2'

**lora_vprediction**: use v-prediction for lora model training

**batch_size**: batch_size or total particle numbers

**particle_num_vsd**: batch size (particle numbers) for VSD training

**particle_num_phi**: number of particles to train phi model

**rgb_as_latents**: initialize particles in latent space

**use_mlp_particle**: use siren mlp as the 2d representation of image

**half_inference**: half-precision inference, requires under 6 GB GPU memory, is faster but has worse performance on vsd

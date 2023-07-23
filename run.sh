#!/bin/sh

# Image generation with prolific_dream 2d 
### VSD
python prolific_dreamer2d.py \
        --num_steps 500 --log_steps 72 \
        --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true --save_phi_model true \

# ### VSD multi particles
# python prolific_dreamer2d.py \
#         --num_steps 1500 --log_steps 50 \
#         --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
#         --model_path 'stabilityai/stable-diffusion-2-1-base' \
#         --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
#         --generation_mode 'vsd' \
#         --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
#         --prompt "a photograph of an astronaut riding a horse" \
#         --height 512 --width 512 --batch_size 16 --guidance_scale 7.5 \
#         --particle_num_vsd 2 --particle_num_phi 2 \
#         --log_progress false --save_x0 false --save_phi_model true

# ### SDS
# python prolific_dreamer2d.py \
#         --num_steps 500 --log_steps 50 --lr 0.03 \
#         --model_path 'stabilityai/stable-diffusion-2-1-base' \
#         --loss_weight '1m_alphas_cumprod' \
#         --t_schedule random --generation_mode 'sds' \
#         --prompt "a photograph of an astronaut riding a horse" \
#         --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
#         --log_progress true --save_x0 true \
#         # --half_inference true

# ### T2I
# python prolific_dreamer2d.py \
#         --num_steps 100 --log_steps 20 --seed 1024 \
#         --model_path 'stabilityai/stable-diffusion-2-1-base' \
#         --generation_mode 't2i' \
#         --prompt "a photograph of an astronaut riding a horse" \
#         --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
#         --log_progress true --save_x0 true \
#         --half_inference true

# ### DreamTime
# python prolific_dreamer2d.py \
#         --num_steps 500 --log_steps 50 --lr 0.03 \
#         --model_path 'stabilityai/stable-diffusion-2-1-base' \
#         --loss_weight 'dreamtime' \
#         --t_schedule dreamtime --generation_mode 'sds' \
#         --prompt "a photograph of an astronaut riding a horse" \
#         --height 512 --width 512 --batch_size 1 --guidance_scale 100 \
#         --log_progress true --save_x0 true \
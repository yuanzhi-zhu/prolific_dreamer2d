#!/bin/sh

### Image generation with prolific_dream 2d 
python prolific_dreamer2d.py \
        --num_steps 500 --t_end 1000 --log_steps 50 --dtype float \
        --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --use_scheduler false --lr_scheduler_start_factor 0.333 --lr_scheduler_iters 200 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction true\
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true --save_phi_model true --multisteps 1 \
        # --load_phi_model_path 'work_dir/20230615_2254_vsd_cfg_7.5_bs_2_lr_0.03_philr_0.0001_lora'

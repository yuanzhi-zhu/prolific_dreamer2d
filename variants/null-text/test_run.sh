#!/bin/sh


# python prolific_dreamer2d.py \
#         --num_steps 500 --log_steps 72 \
#         --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
#         --model_path 'stabilityai/stable-diffusion-2-1-base' \
#         --loss_weight_type '1m_alphas_cumprod' --t_schedule t_stages2 \
#         --generation_mode vsd \
#         --phi_model 'img_variant' --lora_scale 1. \
#         --prompt "a photograph of an astronaut riding a horse" \
#         --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
#         --log_progress true --save_x0 true --save_phi_model true --multisteps 1 \
#         --t_start 20 \
#         --lora_vprediction false \
#         --rgb_as_latents true --use_mlp_particle false \
#         # --particle_num_vsd 2 --particle_num_phi 4
        # --load_phi_model_path 'work_dir/20230615_2254_vsd_cfg_7.5_bs_2_lr_0.03_philr_0.0001_lora'

multisteps=1
t_start=20
lr=0.03
# for grad_scale_phi in 0.9 1.05 1.2 # 1.2 1.5  #0.78 0.73
for grad_scale_phi in 1.0 # 0.8 0.85 0.9 0.95 1.05 1.1 1.15 1.2 # 1.2 1.5  #0.78 0.73
do
    for grad_scale in 1 #0.01 0.1 # -1 # 1.2 1.5  #0.78 0.73
    do
        for phi_update_step in 1 #2 5 10 # 10 15 20 # 2 3 4 5
        do
            for guidance_scale in 7.5 #1 2 3 4 5 8 10
            do
                for num_steps in 500
                do
                    for phi_lr in 0.0002 0.005 0.01 0.02 0.05 0.1 # 0.01 0.02 0.03 #50 #100 300 500
                    do
                        for loss_weight_type in sqrt_alphas_1m_alphas_cumprod # SNR_sqrt SNR_square SNR_log1p # sqrt_alphas_1m_alphas_cumprod
                        do
                            # python back_up/DDS_branch/prolific_dreamer2d.py \
                            python prolific_dreamer2d.py \
                                    --num_steps ${num_steps} --log_steps 72 \
                                    --seed 1024 --lr ${lr} --phi_lr ${phi_lr} --use_t_phi true \
                                    --model_path 'stabilityai/stable-diffusion-2-1-base' \
                                    --loss_weight_type ${loss_weight_type} --t_schedule 'random' \
                                    --generation_mode 'vsd' \
                                    --phi_model 'null_text' --lora_scale 1 --lora_vprediction false \
                                    --prompt "a photograph of an astronaut riding a horse" \
                                    --height 512 --width 512 --batch_size 1 --guidance_scale ${guidance_scale} \
                                    --log_progress true --save_x0 true --save_phi_model true \
                                    --grad_scale_phi ${grad_scale_phi} --grad_scale ${grad_scale} \
                                    --multisteps ${multisteps} --phi_update_step ${phi_update_step} \
                                    # --t_start ${t_start}
                        done
                    done
                done
            done
        done
    done
done


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
# for grad_scale_phi in 0.9 1.05 1.2 # 1.2 1.5  #0.78 0.73
for grad_scale_phi in 1.0 # 0.8 0.85 0.9 0.95 1.05 1.1 1.15 1.2 # 1.2 1.5  #0.78 0.73
do
    for grad_scale in 1 #0.01 0.1 # -1 # 1.2 1.5  #0.78 0.73
    do
        for phi_update_step in 1 # 2 3 4 5
        do
            for guidance_scale in 7.5 #1 2 3 4 5 8 10
            do
                for lr in 0.03
                do
                    for num_steps in 500
                    do
                        python prolific_dreamer2d.py \
                                --num_steps ${num_steps} --log_steps 72 \
                                --seed 1024 --lr ${lr} --phi_lr 0.0001 --use_t_phi true \
                                --model_path 'stabilityai/stable-diffusion-2-1-base' \
                                --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
                                --generation_mode 'vsd' \
                                --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
                                --prompt "a photograph of an astronaut riding a horse" \
                                --height 512 --width 512 --batch_size 1 --guidance_scale ${guidance_scale} \
                                --log_progress true --save_x0 true --save_phi_model true \
                                --grad_scale_phi ${grad_scale_phi} --grad_scale ${grad_scale} \
                                --multisteps ${multisteps} --phi_update_step ${phi_update_step} \
                                # --load_phi_model_path '/cluster/home/jinliang/work/ckpts_yuazhu/prolific_dream2d/work_dir/prolific_dreamer2d_20230706_1821_vsd_cfg_7.5_bs_1_num_steps_20_tschedule_random_lora'
                                # --half_inference true
                    done
                done
            done
        done
    done
done



# lr=0.03
# seed=314
# ### grid search
# for generation_mode in vsd
# do
#     for guidance_scale in 7.5
#     do
#         for grad_scale_phi in 0.8 0.85 # 0.78 0.8 0.85 0.9 0.95 1.05 1.1 1.2 1.5  #0.78 0.73
#         do
#             for t_schedule in random
#             do
#                 for phi_model in lora # img_variant 
#                 do
#                     python prolific_dreamer2d.py \
#                                 --num_steps 1000 --log_steps 72 \
#                                 --seed ${seed} --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
#                                 --loss_weight_type 'none' --t_schedule ${t_schedule} \
#                                 --generation_mode ${generation_mode} \
#                                 --phi_model ${phi_model} --lora_scale 1. \
#                                 --prompt "a photograph of an astronaut riding a horse" \
#                                 --height 512 --width 512 --batch_size 1 --guidance_scale ${guidance_scale} \
#                                 --log_progress true --save_x0 true --save_phi_model true --multisteps 1 \
#                                 --t_start 20 \
#                                 --lora_vprediction false \
#                                 --rgb_as_latents true --use_mlp_particle false \
#                                 --grad_scale_phi ${grad_scale_phi} \
#                                 --model_path 'stabilityai/stable-diffusion-2-1-base' \
#                         #     --num_steps 400 --log_steps 50 \
#                         #     --seed 1024 --phi_lr 0.0001 --use_t_phi true \
#                         #     --model_path 'stabilityai/stable-diffusion-2-1-base' \
#                         #     --loss_weight_type '1m_alphas_cumprod' \
#                         #     --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
#                         #     --prompt "a photograph of an astronaut riding a horse" \
#                         #     --height 512 --width 512 --batch_size 1 \
#                         #     --lr ${lr} \
#                         #     --t_schedule ${t_schedule} \
#                         #     --generation_mode ${generation_mode} \
#                         #     --guidance_scale ${guidance_scale} \
#                         #     --multisteps ${multisteps} \
#                         #     --log_progress true --save_x0 true --save_phi_model true \
#                         #     --rgb_as_latents true --use_mlp_particle false \
#                             # --particle_num_vsd 2 --particle_num_phi 4
#                             # --load_phi_model_path 'work_dir/20230615_2254_vsd_cfg_7.5_bs_2_lr_0.03_philr_0.0001_lora'
#                 done
#             done
#         done
#     done
# done


# ### grid search

# for generation_mode in 'sds' 'vsd'
# do
#     for t_schedule in fixed999 fixed900 fixed800 fixed700 fixed600 fixed500 fixed400 fixed300 fixed200 fixed100
#     do
#         python prolific_dreamer2d.py \
#                 --num_steps 500 --t_end 1000 --log_steps 72 --dtype float \
#                 --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
#                 --use_scheduler false --lr_scheduler_start_factor 0.333 --lr_scheduler_iters 200 \
#                 --model_path 'stabilityai/stable-diffusion-2-1-base' \
#                 --loss_weight_type '1m_alphas_cumprod' --t_schedule ${t_schedule} \
#                 --generation_mode ${generation_mode} \
#                 --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
#                 --prompt "a photograph of an astronaut riding a horse" \
#                 --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
#                 --log_progress true --save_x0 true --save_phi_model true --multisteps 1 \
#     done
# done


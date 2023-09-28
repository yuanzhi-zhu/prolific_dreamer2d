#!/bin/sh

# Image generation with prolific_dream 2d 
### VSD
python prolific_dreamer2d.py \
        --num_steps 500 --log_steps 72 \
        --seed 1024 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'DeepFloyd/IF-I-M-v1.0' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' --multisteps 1 \
        --phi_model 'lora' --lora_scale 1. \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 64 --width 64 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true --save_phi_model true \
        # --use_mlp_particle true \


### SDS
python prolific_dreamer2d.py \
        --num_steps 500 --log_steps 72 \
        --seed 1024 --lr 0.03 \
        --model_path 'DeepFloyd/IF-I-M-v1.0' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 64 --width 64 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        # --use_mlp_particle true \
        # --half_inference true

### T2I pipeline
python prolific_dreamer2d.py \
        --num_steps 100 --log_steps 20 --seed 42 \
        --model_path 'DeepFloyd/IF-I-M-v1.0' \
        --generation_mode 't2i_pipeline' \
        --prompt 'a photograph of an astronaut riding a horse' \
        --height 64 --width 64 --batch_size 1 --guidance_scale 7.5 \
        --log_progress false --save_x0 true \
        # --half_inference true

### T2I
python prolific_dreamer2d.py \
        --num_steps 100 --log_steps 20 --seed 1024 \
        --model_path 'DeepFloyd/IF-I-M-v1.0' \
        --generation_mode 't2i' \
        --prompt 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"' \
        --height 64 --width 64 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        # --half_inference true

       
### T2I pipeline stage 2
python prolific_dreamer2d.py \
        --num_steps 100 --log_steps 20 --seed 42 \
        --model_path 'DeepFloyd/IF-I-M-v1.0' \
        --generation_mode 't2i_stage2' \
        --prompt 'a photograph of an astronaut riding a horse' \
        --height 64 --width 64 --batch_size 1 --guidance_scale 7.5 \
        --log_progress false --save_x0 true \
        --init_img_path 'work_dir/prolific_dreamer2d_20230928_2300_sds_cfg_7.5_bs_1_num_steps_500_tschedule_random/final_image_a_photograph_of_an_astronaut_riding_a_horse.png'
 

#!/bin/bash

###########################################################################################################

n_iters=1000
lr=0.01
# for grad_scale_phi in 0.9 1.05 1.2 # 1.2 1.5  #0.78 0.73
for lr in 0.01 #0.005 0.01 0.02 0.05 0.1 # 2 # 0.8 0.85 0.9 0.95 1.05 1.1 1.15 1.2 # 1.2 1.5  #0.78 0.73
do
    for inner_iter in 1 #2 5 #0.01 0.1 # -1 # 1.2 1.5  #0.78 0.73
    do
        for noisy_input in 1 #0 # 10 15 20 # 2 3 4 5
        do
            # python back_up/DDS_branch/prolific_dreamer2d.py \
            python optimal_test.py \
                    --n_iters ${n_iters} --steps_per_frame 20 \
                    --lr ${lr} --inner_iter ${inner_iter} \
                    --noisy_input ${noisy_input} \
                    --n_samples 64
        done
    done
done


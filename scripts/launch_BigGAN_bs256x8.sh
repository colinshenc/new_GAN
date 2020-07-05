#!/bin/bash



cd /ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/
###conda activate fal_gan_py3.7

#srun
CUDA_VISIBLE_DEVICES=0,1,2,3 /ubc/cs/research/shield/projects/cshen001/anaconda3/envs/fal_gan_py3.7/bin/python train.py \
--dataset C10 --shuffle  --num_workers 12 --batch_size 92  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 20 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_leakyrelu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--dim_z 24576 --shared_dim 128 \
--G_eval_mode --which_best FID\
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--num_epochs 300 --parallel --num_feedback_iter 4 --split_D ###--use_multiepoch_sampler --load_in_mem --G_ch 64 --D_ch 64

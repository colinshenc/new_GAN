#!/bin/bash

#SBATCH --output=/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/slurm_logs/big_gan%j.out
#SBATCH --error=/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/slurm_logs/big_gan%j.err
#SBATCH --nodes=1
###SBATCH --gpus=quadro_rtx_6000:2
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --account=cshen001
#SBATCH --job-name=orig
#SBATCH --cpus-per-task=20
#SBATCH --partition=edith
#SBATCH --time=99999:99:99

cd /ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/
conda activate fal_gan_py3.7

srun /ubc/cs/research/shield/projects/cshen001/anaconda3/envs/fal_gan_py3.7/bin/python train.py \
--dataset C10 --shuffle  --num_workers 20 --batch_size 200  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 20 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_leakyrelu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--dim_z 576 --shared_dim 128 \
--G_eval_mode \
--G_ch 64 --D_ch 64 \
--ema --use_ema --ema_start 20000 \
--test_every 1 --save_every 1 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler --shuffle --num_epochs 1 ###--load_in_mem ####--parallel
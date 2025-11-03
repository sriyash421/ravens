#!/bin/bash
#SBATCH --job-name=ravens_eval
#SBATCH --account=weirdlab 
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=12  
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source ${HOME}/.bashrc
conda activate vpl

ckpt=$1
python rollout.py \
    --checkpoint $ckpt \
    --num_episodes 50 \
    --horizon 400 \
    --video_dir videos/ravens_$2 --continuous $3
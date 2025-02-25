#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name "multi_gpu_denoiser"     # Name of the job
#SBATCH --export=ALL             # Export all environment variables
#SBATCH --output "multi_gpu_attention_error.log"   # Log-file (important!)
#SBATCH --error="multi_gpu_attention.log"
#SBATCH --cpus-per-task=2        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16g       # Memory to allocate in MB per allocated CPU core
#SBATCH --partition=2080ti
#SBATCH --gres=gpu:4             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep_learning

# Train
#python ../preprocess.py
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 multi_gpu_train.py

#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name "DENOISING_TRAIN"     # Name of the job
#SBATCH --export=ALL             # Export all environment variables
#SBATCH --output "out_train.log"   # Log-file (important!)
#SBATCH --error="error_train.log"
#SBATCH --cpus-per-task=1        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
#SBATCH --partition=1080ti
#SBATCH --gres=gpu:2             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time
#


# Activate the Anaconda environment in which to execute the Jupyter instance.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep_learning

# Start Jupyter Lab
python ../train.py
#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name "JUPYTER"     # Name of the job 
#SBATCH --export=ALL             # Export all environment variables
#SBATCH --output "jupyter.log"   # Log-file (important!)
#SBATCH --cpus-per-task=1        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
#SBATCH --partition=1080ti
#SBATCH --gres=gpu:1             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time
#

# Activate the Anaconda environment in which to execute the Jupyter instance.
# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep_learning
# Start Jupyter Lab
jupyter notebook --ip=0.0.0.0 --port=8012   --no-browser

#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name "JUPYTER"     # Name of the job 
#SBATCH --export=ALL             # Export all environment variables
#SBATCH --output "jupyter.log"   # Log-file (important!)
#SBATCH --cpus-per-task=1        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
#SBATCH --partition=2080ti
#SBATCH --gres=gpu:2             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time
#

# Activate the Anaconda environment in which to execute the Jupyter instance.
conda activate deep_learning # CHANGEME

# Start Jupyter Lab
jupyter notebook --ip=0.0.0.0 --no-browser

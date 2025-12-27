#!/bin/bash
#SBATSH --job-name=llm_test3
#SBATCH --time=20:00:00
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1        # CHANGED: Set to 1 so srun runs the script once
#SBATCH --cpus-per-task=10         # OPTIONAL: Increase CPUs to help data loading
#SBATCH --mem=50G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export TORCH_NUM_THREADS=10

# Run directly with python. 
# DataParallel will automatically detect the 2 GPUs provided by --gres=gpu:2
python -u TermTyping3.py
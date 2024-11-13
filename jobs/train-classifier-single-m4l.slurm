#!/bin/bash -l
# Standard output and error:
#SBATCH -o ../logs/job.out.%j
#SBATCH -e ../logs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J c6_4l_clf_single_m4l
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=40000

#SBATCH --mail-type=none
#SBATCH --mail-user=griesemx@mpp.mpg.de
#SBATCH --time=08:00:00

# #SBATCH --partition=gpudev

module purge
module load python-waterboa/2024.06

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PYTHONUNBUFFERED=TRUE

source ../../bsc-thesis/venv/bin/activate
srun python train-classifier-single-m4l.py
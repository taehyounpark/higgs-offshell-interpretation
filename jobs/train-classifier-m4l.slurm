#!/bin/bash -l
# Standard output and error:
#SBATCH -o ../logs/job.out.%j
#SBATCH -e ../logs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J c6_4l_clf_m4l
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=50000

#SBATCH --mail-type=none
#SBATCH --mail-user=griesemx@mpp.mpg.de
#SBATCH --time=24:00:00

# #SBATCH --partition=gpudev

module purge
module load python-waterboa/2024.06 cuda/12.6

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PYTHONUNBUFFERED=TRUE

source ../venv/bin/activate
srun python train-classifier-m4l.py
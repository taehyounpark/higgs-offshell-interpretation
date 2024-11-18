#!/bin/bash -l
# Standard output and error:
#SBATCH -o ../logs/job.out.%j
#SBATCH -e ../logs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J plot_predictions
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100000

#SBATCH --mail-type=none
#SBATCH --mail-user=griesemx@mpp.mpg.de
#SBATCH --time=24:00:00

# #SBATCH --partition=gpudev

module purge
module load anaconda/3/2023.03 
module load tensorflow/gpu-cuda-12.1/2.14.0 protobuf/4.24.0 mkl/2023.1 cuda/12.1 cudnn/8.9.2 nccl/2.18.3 tensorrt/8.6.1 tensorboard/2.13.0 keras/2.14.0 keras-preprocessing/1.1.2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PYTHONUNBUFFERED=TRUE

source ../../venv/bin/activate
srun python plot-pred.py
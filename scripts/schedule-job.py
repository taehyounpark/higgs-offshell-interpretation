import os
import subprocess
import shutil
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description='Python script for scheduling slurm jobs on a cluster.')
    parser.add_argument('command', type=str, help='Command to be run in the SLURM job.')
    parser.add_argument('-p', '--prep-command', type=str, help='Command to prepare the job environment.')
    parser.add_argument('-r', '--runstring', type=str, help='The name for the job, listed in SLURM queue and will be the name of the output directory.', required=True)
    parser.add_argument('-d', '--run-dir', type=str, help='The directory where the job output folder will be placed.', required=True)
    parser.add_argument('-g', '--num-gpus', type=int, default=1, help='Number of GPUs required by the job.')
    parser.add_argument('-c', '--num-cores', type=int, default=4, help='Number of CPU cores required by the job.')
    parser.add_argument('-m', '--mem', type=int, default=16000, help='Memory in kB required by the job.')
    parser.add_argument('-t', '--time', type=str, default='06:00:00', help='Maximum runtime for the job. Format HH:MM:SS')

    args = parser.parse_args()
    return args


def write_job(args):
    os.makedirs(os.path.join(args.run_dir, args.runstring), exist_ok=True)
    
    script_contents = f"""#!/bin/bash -l
#SBATCH -o {os.path.join(args.run_dir, args.runstring)}/logs/job.out.%j
#SBATCH -e {os.path.join(args.run_dir, args.runstring)}/logs/job.err.%j
#SBATCH -D ./
#SBATCH -J {args.runstring}
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:{args.num_gpus}
#SBATCH --cpus-per-task={args.num_cores}
#SBATCH --mem={args.mem}
#SBATCH --mail-type=none
#SBATCH --mail-user=griesemx@mpp.mpg.de
#SBATCH --time={args.time}
# #SBATCH --partition=gpudev

module purge
module load anaconda/3/2023.03 
module load tensorflow/gpu-cuda-12.1/2.14.0 protobuf/4.24.0 mkl/2023.1 cuda/12.1 cudnn/8.9.2 nccl/2.18.3 tensorrt/8.6.1 tensorboard/2.13.0 keras/2.14.0 keras-preprocessing/1.1.2

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

export PYTHONUNBUFFERED=TRUE

source ../../venv/bin/activate
{args.command}
"""
    script_path = f"{os.path.join(args.run_dir, args.runstring)}/job.slurm"
    with open(script_path, 'w') as script_file:
        script_file.write(script_contents)
    return script_path

def submit_job(args, script_path):
    try:
        subprocess.run(f"sbatch {script_path}", stderr=subprocess.STDOUT, shell=True, check=True)
        print(f'Submitted job {args.runstring}')
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {args.runstring}: {e}")

def prepare_rundir(args):
    if args.prep_command is not None:
        try:
            #TODO: Find some better way of doing this
            cmd = args.prep_command.split(' ')

            prep_cmd = cmd[:2]
            prep_cmd.append(os.path.join(args.run_dir, args.runstring))
            prep_cmd.extend(cmd[2:])
            
            subprocess.run(' '.join(prep_cmd), stderr=subprocess.STDOUT, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to run {args.prep_command}: {e}")

def main(args):
    slurm_script_path = write_job(args)
    prepare_rundir(args)
    submit_job(args, slurm_script_path)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
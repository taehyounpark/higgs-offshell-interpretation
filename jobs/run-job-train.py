import os
import subprocess
import shutil


def write_job(job):
    runstring, command, params = job
    os.makedirs(runstring, exist_ok=True)
    
    script_contents = f"""#!/bin/bash -l
#SBATCH -o {runstring}/logs/job.out.%j
#SBATCH -e {runstring}/logs/job.err.%j
#SBATCH -D ./
#SBATCH -J {runstring}
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:{params['n-gpus']}
#SBATCH --cpus-per-task={params['n-cpus']}
#SBATCH --mem={params['mem']}
#SBATCH --mail-type=none
#SBATCH --mail-user=griesemx@mpp.mpg.de
#SBATCH --time={params['time']}
# #SBATCH --partition=gpudev

module purge
module load anaconda/3/2023.03 
module load tensorflow/gpu-cuda-12.1/2.14.0 protobuf/4.24.0 mkl/2023.1 cuda/12.1 cudnn/8.9.2 nccl/2.18.3 tensorrt/8.6.1 tensorboard/2.13.0 keras/2.14.0 keras-preprocessing/1.1.2

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

export PYTHONUNBUFFERED=TRUE

source ../../venv/bin/activate
{command}
"""
    script_path = f"{runstring}/job.slurm"
    with open(script_path, 'w') as script_file:
        script_file.write(script_contents)
    return script_path

def submit_job(job):
    runstring, _, _ = job
    script_path = write_job(job)

    shutil.copyfile('train-nn.py', f'{runstring}/execute.py')
    
    try:
        subprocess.run(f"sbatch {script_path}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {runstring}: {e}")

def define_job(runstring, slurm_params, train_params, train_flags=[]):
    command = f'python {runstring}/execute.py '

    train_params_all = ['learn-rate', 'batch-size', 'epochs', 'num-events', 'num-layers', 'num-nodes', 'sample-dir', 'c6']
    train_flags_all = ['sig','int','sig-vs-sbi','int-vs-sbi','bkg-vs-sbi']

    for flag in train_flags:
        if flag in train_flags_all:
            command += '--' + flag + ' '

    for key, value in train_params.items():
        if value is not None and key in train_params_all:
            command += '--' + key + '=' + str(value).replace('[','').replace(']','').replace('(','').replace(')','').replace(' ', '') + ' '

    command += '-o ' + runstring
    
    slurm_params['n-gpus'] = 1 if 'n-gpus' not in slurm_params.keys() else slurm_params['n-gpus']
    slurm_params['n-cpus'] = 6 if 'n-cpus' not in slurm_params.keys() else slurm_params['n-cpus']
    slurm_params['mem'] = 16000 if 'mem' not in slurm_params.keys() else slurm_params['mem']
    slurm_params['time'] = '01:00:00' if 'time' not in slurm_params.keys() else slurm_params['time']

    return (runstring, command, slurm_params)

def main():
    joblist = [
        define_job('train-multi-SIG', 
                   slurm_params={'time': '16:00:00', 'n-cpus': 36, 'n-gpus': 2, 'mem': 60000}, 
                   train_flags=['sig'], 
                   train_params={'num-events': 1000, 'c6': [-20,20,2001], 'epochs': 120, 'batch-size': 32, 'learning-rate': 1e-5}),
        define_job('train-multi-SBI', 
                   slurm_params={'time': '16:00:00', 'n-cpus': 36, 'n-gpus': 2, 'mem': 60000},
                   train_params={'num-events': 1000, 'c6': [-20,20,2001], 'epochs': 120, 'batch-size': 32, 'learning-rate': 1e-5})
    ]

    joblist = [
        define_job('train-SIG-vs-SBI',
                   slurm_params={'time': '24:00:00', 'n-cpus': 18, 'n-gpus': 1, 'mem': 60000},
                   train_flags=['sig-vs-sbi'],
                   train_params={'num-events': 100000, 'c6': [-20,20,11], 'epochs': 120, 'batch-size': 32, 'learning-rate': 1e-5}),
        define_job('train-INT-vs-SBI',
                   slurm_params={'time': '24:00:00', 'n-cpus': 18, 'n-gpus': 1, 'mem': 60000},
                   train_flags=['int-vs-sbi'],
                   train_params={'num-events': 100000, 'c6': [-20,20,11], 'epochs': 120, 'batch-size': 32, 'learning-rate': 1e-5}),
        define_job('train-BKG-vs-SBI',
                    slurm_params={'time': '16:00:00', 'n-cpus': 18, 'n-gpus': 1, 'mem': 60000},
                    train_params={'num-events': 100000, 'epochs':80, 'batch-size': 32, 'learning-rate':1e-5})
    ]

    for job in joblist:
        submit_job(job)
        print('Starting job', job)

    #job = define_job('train-INT-vs-SBI',
    #               slurm_params={'time': '12:00:00', 'n-cpus': 36, 'n-gpus': 4, 'mem': 100000},
    #               train_flags=['int-vs-sbi'],
    #               train_params={'num-events': 100000, 'c6': [-20,20,101], 'epochs': 120, 'batch_size': 32, 'learning_rate': 1e-5})
    #submit_job(job)

if __name__ == '__main__':
    main()
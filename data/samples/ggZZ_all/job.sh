#!/usr/bin/env bash
#SBATCH --job-name=ggZZ_all
#SBATCH --output=ggZZ_all/%j.out
#SBATCH --error=ggZZ_all/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ./setup.sh
./mcfm ./input_ggZZ_all.ini -general%rundir=ggZZ_all -general%runstring=ggZZ_all 

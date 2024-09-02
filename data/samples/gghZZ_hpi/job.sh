#!/usr/bin/env bash
#SBATCH --job-name=gghZZ_hpi
#SBATCH --output=gghZZ_hpi/%j.out
#SBATCH --error=gghZZ_hpi/%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ./setup.sh
./mcfm ./input_gghZZ_hpi.ini -general%rundir=gghZZ_hpi -general%runstring=gghZZ_hpi 

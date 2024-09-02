#!/usr/bin/env bash
#SBATCH --job-name=gghZZ_int
#SBATCH --output=gghZZ_int/%j.out
#SBATCH --error=gghZZ_int/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ./setup.sh
./mcfm ./input_gghZZ_int.ini -general%rundir=gghZZ_int -general%runstring=gghZZ_int 

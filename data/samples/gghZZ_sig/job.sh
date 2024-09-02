#!/usr/bin/env bash
#SBATCH --job-name=gghZZ_sig
#SBATCH --output=gghZZ_sig/%j.out
#SBATCH --error=gghZZ_sig/%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ./setup.sh
./mcfm ./input_gghZZ_sig.ini -general%rundir=gghZZ_sig -general%runstring=gghZZ_sig 

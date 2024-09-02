#!/usr/bin/env bash
#SBATCH --job-name=ggZZ_box
#SBATCH --output=ggZZ_box/%j.out
#SBATCH --error=ggZZ_box/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ./setup.sh
./mcfm ./input_ggZZ_box.ini -general%rundir=ggZZ_box -general%runstring=ggZZ_box 

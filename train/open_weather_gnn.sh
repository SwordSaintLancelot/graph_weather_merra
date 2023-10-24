#!/bin/bash
#SBATCH --mail-user=vgaur@nsstc.uah.edu  
#SBATCH -J FCN
#SBATCH --gres=gpu:a100
#SBATCH --ntasks 2
#SBATCH -t 20-00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR

########## Add all commands below here

export PATH="/rhome/vgaur/miniconda3/envs/graph-weather/bin:$PATH"

module load cuda

python run_fulll_dsig.py 
echo '**********Finished!**********'

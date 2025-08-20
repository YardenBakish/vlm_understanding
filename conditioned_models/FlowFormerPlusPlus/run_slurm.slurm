#! /bin/sh

#SBATCH --account=gpu-research
#SBATCH --job-name=train_sintel
#SBATCH --output=slurm_logs/train_sintel.out # redirect stdout
#SBATCH --error=slurm_logs/train_sintel.err # redirect stderr
#SBATCH --partition=gpu-ai # (see resources section)
#SBATCH --time=720 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=10000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python -u train_FlowFormer.py --name sintel --stage sintel --validation sintel
#! /bin/sh


#SBATCH --job-name=trains
#SBATCH --output=slurm_logs/train.out # redirect stdout
#SBATCH --error=slurm_logs/train.err # redirect stderr
#SBATCH --partition=killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=10000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python create_teacher_data.py --mode create_tracks --partition 0
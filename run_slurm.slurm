#! /bin/sh

#SBATCH --job-name=train_opt           # Job name
#SBATCH --account=gpu-tad-users # Account name
#SBATCH --partition=gpu-tad     # Partition name
#SBATCH --time=23:59:00               # Max run time (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --mem-per-cpu=50G              # Memory per CPU
#SBATCH --output=slurm_logs/train_opt.out        # Output file
#SBATCH --error=slurm_logs/train_opt.err         # Error file
#SBATCH --gres=gpu:1 




python distillation.py --mode optFlow_extended
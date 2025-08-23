#! /bin/sh

#SBATCH --job-name=p6            # Job name
#SBATCH --account=gpu-tad-users # Account name
#SBATCH --partition=gpu-tad     # Partition name
#SBATCH --time=12:00:00               # Max run time (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --mem-per-cpu=100G              # Memory per CPU
#SBATCH --output=slurm_logs/p6.out        # Output file
#SBATCH --error=slurm_logs/p6.err         # Error file
#SBATCH --gres=gpu:1 




python create_teacher_data.py --mode create_tracks --partition 6
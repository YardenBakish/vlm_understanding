#! /bin/sh

#SBATCH --job-name=2           # Job name
#SBATCH --account=gpu-tad-users # Account name
#SBATCH --partition=gpu-tad     # Partition name
#SBATCH --time=00:59:00               # Max run time (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=8             # CPUs per task
#SBATCH --mem-per-cpu=50G              # Memory per CPU
#SBATCH --output=slurm_logs/2.out        # Output file
#SBATCH --error=slurm_logs/2.err         # Error file
#SBATCH --gres=gpu:4 


export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=$(nvidia-smi -L | wc -l)


python -m torch.distributed.launch \
    --nproc_per_node=$(nvidia-smi -L | wc -l) \
    --master_port=$MASTER_PORT \
    train_internvl_w_cfg.py --mode joint_learning_extended_mask_freeze
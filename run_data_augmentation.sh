#!/bin/bash
#SBATCH --job-name="Data Augmentation"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --partition=gpu-invest

# enable this when on gpu partition (and NOT on gpu-invest)
###SBATCH --qos=job_gpu_preempt

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate data_aug

# Put your code below this line
python -m data_augmentation.translator

# IMPORTANT:
# Run with                  sbatch run_data_augmentation.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=128G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash

#!/bin/bash
#SBATCH --job-name="Swiss Judgement Prediction"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=none
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00

# Put your code below this line
bash run.sh > run.out

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sl
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash



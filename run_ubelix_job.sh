#!/bin/bash
#SBATCH --job-name="Swiss Judgement Prediction"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --array=1-5%1

# Put your code below this line
# $1: model_name, $2: type, $3: lang
bash run.sh -m $1 -t $2 -l $3 -s ${SLURM_ARRAY_TASK_ID} -d False >current-run.out

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sl
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash

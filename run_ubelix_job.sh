#!/bin/bash
#SBATCH --job-name="SJP"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu
#SBATCH --array=1-5%1

# enable this when on gpu partition (and NOT on gpu-invest)
###SBATCH --qos=job_gpu_preempt


# Put your code below this line

# $1: model_name, $2: type, $3: language, $4: train_language, $5: mode, $6 special_splits
bash run.sh --model_name=$1 --type=$2 --language=$3 --train_language=$4 --mode=$5 --special_splits=$6 \
  --seed=${SLURM_ARRAY_TASK_ID} --debug=False >current-run.out

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sl
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash

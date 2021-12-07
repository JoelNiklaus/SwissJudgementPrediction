#!/bin/bash
#SBATCH --job-name="SJP"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu
#SBATCH --array=1-5%5

# enable this when on gpu partition (and NOT on gpu-invest)
###SBATCH --qos=job_gpu_preempt

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate sjp

# Put your code below this line
#           $1: train_type, $2: train_mode, $3: model_name, $4: model_type, $5: train_languages, $6: test_languages, $7: sub_datasets
bash run.sh --train_type=$1 --train_mode=$2 --model_name=$3 --model_type=$4 --train_languages=$5 --test_languages=$6 --sub_datasets=$7 \
  --seed=${SLURM_ARRAY_TASK_ID} --debug=False >current-run.out

# Example: bash run.sh --train_type=adapters --train_mode=train --model_name=xlm-roberta-base --model_type=hierarchical --train_languages=it --test_languages=it  --sub_datasets=False --seed=1 --debug=True
# Example: sbatch run_ubelix_job.sh adapters train xlm-roberta-base hierarchical de de False

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash

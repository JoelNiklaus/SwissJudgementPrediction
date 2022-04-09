#!/bin/bash
#SBATCH --job-name="SJP"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=20-00:00:00
#SBATCH --qos=job_gpu_stuermer
#SBATCH --partition=gpu-invest
#SBATCH --array=2-4

# enable this for multiple GPUs for max 24h
###SBATCH --time=24:00:00
###SBATCH --qos=job_gpu_preempt
###SBATCH --partition=gpu

# enable this to get a time limit of 20 days
###SBATCH --time=20-00:00:00
###SBATCH --qos=job_gpu_stuermer
###SBATCH --partition=gpu-invest

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate sjp

# Put your code below this line
#           $1: train_type, $2: train_mode, $3: model_name, $4: model_type, $5: train_languages, $6: test_languages,$7: jurisdiction, $8: data_augmentation_type, $9: train_sub_datasets ${10}: sub_datasets
bash run.sh --train_type=$1 --train_mode=$2 --model_name=$3 --model_type=$4 --train_languages=$5 --test_languages=$6 --jurisdiction=$7 --data_augmentation_type=$8 --train_sub_datasets=$9 --sub_datasets=${10} \
  --seed=${SLURM_ARRAY_TASK_ID} --debug=False >current-run.out

# Example: bash run.sh --train_type=adapters --train_mode=train --model_name=xlm-roberta-base --model_type=hierarchical --train_languages=it --test_languages=it --jurisdiction=switzerland --data_augmentation_type=no_augmentation --train_sub_datasets=civil_law --sub_datasets=False --seed=1 --debug=True
# Example: sbatch run_ubelix_job.sh adapters train xlm-roberta-base hierarchical de de switzerland no_augmentation civil_law False

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash

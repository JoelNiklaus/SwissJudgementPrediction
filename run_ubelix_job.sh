#!/bin/bash
#SBATCH --job-name="Swiss Judgement Prediction"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --array=1-5%1

# German models
#MODEL_NAME="deepset/gbert-base"

# French models
MODEL_NAME="camembert/camembert-base-ccnet"

# Italian models
#MODEL_NAME="dbmdz/bert-base-italian-cased"

# Multilingual models
#MODEL_NAME="distilbert-base-multilingual-cased"
#MODEL_NAME="bert-base-multilingual-cased"
#MODEL_NAME="xlm-roberta-base"
#MODEL_NAME="xlm-roberta-large"

TYPE='longformer' # 'standard', 'long', 'longformer', 'hierarchical'
LANG='fr'         #'de', 'fr', 'it'

# Put your code below this line
bash run.sh -m $MODEL_NAME -t $TYPE -l $LANG -s ${SLURM_ARRAY_TASK_ID} >current-run.out

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sl
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --time=02:00:00 --pty /bin/bash

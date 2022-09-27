#!/bin/bash
#SBATCH --job-name="Data Generalization"
###SBATCH --mail-user=
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu
#SBATCH --partition=gpu

# enable this when on gpu partition (and NOT on gpu-invest)
###SBATCH --qos=job_gpu_preempt
# enable this to get a time limit of 20 days
###SBATCH --qos=job_gpu_stuermer

# alternatively run multiprocess on 6 gtx1080ti gpus with qos job_gpu_preempt (further reduce batch size): only works with opus-mt model

# Activate correct conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate data_aug

# Put your code below this line
python -m data_generalization.date_normalizer

# IMPORTANT:
# Run with                  sbatch run_data_generalization.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=128G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash
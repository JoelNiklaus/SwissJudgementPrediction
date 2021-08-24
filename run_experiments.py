import pandas as pd
import subprocess

mode = "test"
sub_datasets = True
file_name = f"{mode}_experiments.csv"

df = pd.read_csv(file_name)
for index, row in df.iterrows():
    command = f"sbatch run_ubelix_job.sh {row['model_name']} {row['type']} {row['lang']} {row['lang']} {mode} {sub_datasets}"
    rc = subprocess.call(command, shell=True)
    df.loc[index, "status"] = "started"
    df.to_csv(file_name)

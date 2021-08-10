import pandas as pd
import subprocess

df = pd.read_csv("train_experiments.csv")

for index, row in df.iterrows():
    rc = subprocess.call(f"run_ubelix_job.sh {row['model_name']} {row['type']} {row['lang']}", shell=True)
    df.loc[index, "status"] = "started"
    df.to_csv("train_experiments.csv")

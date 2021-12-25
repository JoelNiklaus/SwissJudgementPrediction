import pandas as pd
import wandb


def retrieve_results(project_name, user="joelito"):
    """Retrieve the results from the wandb api, save them to a csv file and return as a df"""
    api = wandb.Api()
    # Project is specified by <entity/project-name>

    runs = api.runs(f"{user}/{project_name}")
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .name is the human-readable name of the run.
        name_list.append(run.name)

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        # We remove the gradients
        summary_list.append({k: v for k, v in run.summary._json_dict.items()
                             if not k.startswith('gradients')})

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})
    df = pd.DataFrame({
        "name": name_list,
        "summary": summary_list,
        "config": config_list,
    })
    df.to_csv("project.csv")  # save all results to csv for manual inspection
    return df


def pd_dp(df):
    """
    pd_dp for "pandas debug print". Prints a df in its entire length
    :param df:
    :return:
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
        print(df)

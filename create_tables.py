import pandas as pd

from arguments.data_arguments import DataAugmentationType
from arguments.model_arguments import TrainType, LongInputBertType
from utils.wandb_util import retrieve_results, result_cell, pd_dp


class Experiment:
    name = "experiment"
    num_random_seeds = 3
    model_types = [LongInputBertType.HIERARCHICAL]
    train_types = [TrainType.FINETUNE]
    data_augmentation_types = [DataAugmentationType.NO_AUGMENTATION]
    train_langs = ['de', 'fr', 'it']
    test_langs = ['de', 'fr', 'it']
    bert_models = {
        "de": "deepset/gbert-base",
        "fr": "camembert/camembert-base-ccnet",
        "it": "Musixmatch/umberto-commoncrawl-cased-v1",
        "en": "roberta-base",
        "ml": "xlm-roberta-base"  # multilingual
    }


class MonolingualExperiment(Experiment):
    name = "monolingual"
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]


class MultilingualExperiment(Experiment):
    name = "multilingual"
    train_langs = ['ml']
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]


def create_table(df: pd.DataFrame, experiment: Experiment):
    """Creates a table based on the reults and the experiment config"""
    table = []
    for train_lang in experiment.train_langs:
        for train_type in experiment.train_types:
            for model_type in experiment.model_types:
                run_name = f'{train_type}-{experiment.bert_models[train_lang]}-{model_type}-{train_lang}'
                lang_df = df[df.name.str.contains(run_name)]  # filter by run_name
                # TODO maybe add another config param for experiment name for easy filtering
                for data_augmentation_type in experiment.data_augmentation_types:
                    filter = lambda x: x['data_args']['data_augmentation_type'] == data_augmentation_type
                    da_df = lang_df[lang_df.config.apply(filter)]  # create data_augmentation_df
                    # pd_dp(da_df.summary)
                    # if this fails, there might be some failed/crashed runs which need to be deleted
                    assert len(da_df.index) == experiment.num_random_seeds

                    table.append(get_row(experiment, run_name, da_df))  # add results per experiment row

    table_df = pd.DataFrame(table)
    table_df = table_df.set_index("row_desc")
    pd_dp(table_df)
    table_df.to_latex(f"{experiment.name}.tex")


def get_row(experiment, run_name, lang_df):
    row_dict = {"row_desc": run_name}
    for test_lang in experiment.test_langs:
        try:
            f1_macro = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/f1_macro'])
        except KeyError:
            continue  # skip test lang if the test result is not present
        row_dict[test_lang] = result_cell(f1_macro)  # add results per language
    return row_dict


project_name = "SwissJudgmentPredictionCrossLingualTransfer"
original_df = retrieve_results(project_name)

create_table(original_df, MonolingualExperiment())
create_table(original_df, MultilingualExperiment())

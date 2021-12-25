from typing import List

import pandas as pd
import numpy as np

from arguments.data_arguments import DataAugmentationType
from arguments.model_arguments import TrainType, LongInputBertType
from utils.wandb_util import retrieve_results, pd_dp

import re


class RegexDict(dict):
    """
    Special dict for searching keys with regex:
    https://stackoverflow.com/questions/21024822/python-accessing-dictionary-with-wildcards
    """

    def get_matching_keys(self, event):
        return (key for key in self if re.match(event, key))


def get_bert_models(language: str, with_xlm_r=True, with_m_bert=False):
    models = []
    if with_xlm_r:
        models.append("xlm-roberta-base")
    if with_m_bert:
        models.append("bert-base-multilingual-cased")
    if language == 'de':
        models.append("deepset/gbert-base")
    if language == 'fr':
        models.append("camembert/camembert-base-ccnet")
    if language == 'it':
        models.append("Musixmatch/umberto-commoncrawl-cased-v1")
    if language == 'en':
        models.append("roberta-base")
    return models


class Experiment:
    name = "experiment"
    num_random_seeds = 3
    model_types = [LongInputBertType.HIERARCHICAL]
    train_types = [TrainType.FINETUNE]
    data_augmentation_types = [DataAugmentationType.NO_AUGMENTATION]
    train_langs = ['de', 'fr', 'it', 'en']
    test_langs = ['de', 'fr', 'it', 'en']


class MonoLingualExperiment(Experiment):
    name = "mono-lingual"
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]


class ZeroShotCrossLingualExperiment(Experiment):
    name = "zero-shot-cross-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    train_langs = ['de,fr', 'de,it', 'fr,it']
    test_langs = ['de', 'fr', 'it']


class MultiLingualExperiment(Experiment):
    name = "multi-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    train_langs = ['de,fr,it']
    test_langs = ['de', 'fr', 'it']


class ResultCell:
    def __init__(self, mean=0, std=0, connector='±'):
        self.mean = mean
        self.std = std
        self.connector = connector

    def round(self, num):
        """Convert a result from 0-1 to a number between 0 and 100 rounded to 2 decimals"""
        return (num * 100).round(2)

    def __str__(self):
        return f"{self.round(self.mean)} {self.connector} {self.round(self.std)}"


def create_table(df: pd.DataFrame, experiment: Experiment):
    """Creates a table based on the results and the experiment config"""
    table = []
    for train_lang in experiment.train_langs:
        for train_type in experiment.train_types:
            for model_type in experiment.model_types:
                for model in get_bert_models(train_lang):
                    run_name = f'{train_type}-{model}-{model_type}-{train_lang}-'
                    lang_df = df[df.name.str.contains(run_name)]  # filter by run_name
                    # TODO maybe add another config param for experiment name for easy filtering
                    for data_augmentation_type in experiment.data_augmentation_types:
                        filter = lambda x: x['data_args']['data_augmentation_type'] == data_augmentation_type
                        da_df = lang_df[lang_df.config.apply(filter)]  # create data_augmentation_df
                        if len(da_df.index) > 0:
                            # pd_dp(da_df.summary)
                            # if this fails, there might be some failed/crashed runs which need to be deleted
                            assert len(da_df.index) == experiment.num_random_seeds

                            row_name = run_name + " " + data_augmentation_type
                            table.append(get_row(experiment, row_name, da_df))  # add results per experiment row

    table_df = pd.DataFrame(table)
    table_df = table_df.set_index("row_desc")
    pd_dp(table_df)
    table_df.to_latex(f"experiment_{experiment.name}.tex")


# TODO if we want to get a single number for the sub datasets: what do we want to measure? do we want to measure the difference in performance between one dataset to another?
def get_row(experiment, run_name, lang_df, sub_datasets=None, metric="f1_macro"):
    """get the results from sub datasets and the test sets of each language"""
    if sub_datasets is None:  # default argument for sub_datasets
        sub_datasets = ['year', 'input_length', 'legal_area', 'origin_region', 'origin_canton']
    row_dict = {"row_desc": run_name}
    # only consider first entry, the others should have the same structure
    reg_dict = RegexDict(lang_df.iloc[0].summary)
    lang_result_cells = []
    sd_lang_result_cells = []
    for test_lang in experiment.test_langs:
        matches = reg_dict.get_matching_keys(f'test/{test_lang}/{metric}')
        if next(matches, -1) == -1:  # if we did not find any of entries in that language
            continue  # skip it
        sd_class_result_cells = []
        for sub_dataset in sub_datasets:
            keys = reg_dict.get_matching_keys(f'{test_lang}/{sub_dataset}/.+/{metric}')
            sd_instance_result_cells = []
            for key in keys:
                # compute average over all instances of a sub dataset ==> e.g. year: 2017, .., 2020
                sd_instance_scores = lang_df.summary.apply(lambda x: x[key])  # series over random seeds
                sd_instance_result_cells.append(ResultCell(sd_instance_scores.mean(), sd_instance_scores.std()))
            # compute average over all sub dataset classes of a language ==> e.g. de/year
            row_dict[f"sd-{test_lang}-{sub_dataset}"] = aggregate_result_cells(sd_instance_result_cells)
            sd_class_result_cells.append(row_dict[f"sd-{test_lang}-{sub_dataset}"])
        # compute average over all sub dataset languages ==> e.g. de
        row_dict[f"sd-{test_lang}"] = aggregate_result_cells(sd_class_result_cells)
        sd_lang_result_cells.append(row_dict[f"sd-{test_lang}"])

        lang_scores = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/{metric}'])  # series over random seeds
        row_dict[f"lang-{test_lang}"] = ResultCell(lang_scores.mean(), lang_scores.std())  # add results per language
        lang_result_cells.append(row_dict[f"lang-{test_lang}"])
    row_dict['lang-avg'] = aggregate_result_cells(lang_result_cells)  # average of languages
    row_dict['sd-avg'] = aggregate_result_cells(sd_lang_result_cells)  # average of languages sub datasets

    return row_dict


def aggregate_result_cells(result_cells: List[ResultCell]) -> ResultCell:
    """aggregates a list of result cells into another result cell"""
    mean = np.mean([rc.mean for rc in result_cells])
    std = np.mean([rc.std for rc in result_cells])
    return ResultCell(mean, std)


project_name = "SwissJudgmentPredictionCrossLingualTransfer"
original_df = retrieve_results(project_name)

create_table(original_df, MonoLingualExperiment())
create_table(original_df, ZeroShotCrossLingualExperiment())
create_table(original_df, MultiLingualExperiment())

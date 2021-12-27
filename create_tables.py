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


display_names = {
    "xlm-roberta-base": "XLM-R",
    "bert-base-multilingual-cased": "mBERT",
    "deepset/gbert-base": "NativeBERT",
    "camembert/camembert-base-ccnet": "NativeBERT",
    "Musixmatch/umberto-commoncrawl-cased-v1": "NativeBERT",
    "roberta-base": "NativeBERT",
    TrainType.FINETUNE: "FT",
    TrainType.ADAPTERS: "AD",
    DataAugmentationType.NO_AUGMENTATION: "no-aug",
    DataAugmentationType.TRANSLATION: "trans",
}


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
    def __init__(self, mean=0, std=0, min=0, max=0, connector='±', empty=False):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.connector = connector
        self.empty = empty  # we got now result

    def round(self, num):
        """Convert a result from 0-1 to a number between 0 and 100 rounded to 2 decimals"""
        return (num * 100).round(2)

    def __str__(self):
        if self.empty:
            return "–"
        return f"{self.round(self.mean)} {self.connector} {self.round(self.std)} " \
               f"({self.round(self.min)} - {self.round(self.max)})"


def create_table(df: pd.DataFrame, experiment: Experiment,
                 show_sub_datasets_results=False, filter_out_train_sub_datasets=True):
    """Creates a table based on the results and the experiment config"""
    table = {}
    for train_lang in experiment.train_langs:
        for train_type in experiment.train_types:
            for model_type in experiment.model_types:
                for model in get_bert_models(train_lang):
                    run_name = f'{train_type}-{model}-{model_type}-{train_lang}-'
                    lang_df = df[df.name.str.contains(run_name)]  # filter by run_name
                    if filter_out_train_sub_datasets:
                        lang_df = lang_df[lang_df.config.apply(lambda x: 'train_sub_datasets' not in x['data_args'])]
                    # TODO maybe add another config param for experiment name for easy filtering
                    for data_augmentation_type in experiment.data_augmentation_types:
                        filter = lambda x: x['data_args']['data_augmentation_type'] == data_augmentation_type
                        da_df = lang_df[lang_df.config.apply(filter)]  # create data_augmentation_df
                        if len(da_df.index) > 0:
                            # pd_dp(da_df.summary)
                            # if this fails, there might be some failed/crashed runs which need to be deleted
                            assert len(da_df.index) == experiment.num_random_seeds

                            row_name = f"{display_names[model]} " \
                                       f"{display_names[train_type]} " \
                                       f"{display_names[data_augmentation_type]}"
                            # add results per experiment row: merge dicts with same row name (e.g. NativeBERTs)
                            table[row_name] = {**(table[row_name] if row_name in table else {}),
                                               **(get_row(experiment, da_df))}

    # only compute lang-avg and sd-avg across languages here because NativeBERTs are not all available in get_row()
    for row_name in table.keys():
        lang_result_cells = [table[row_name][f"lang-{test_lang}"] for test_lang in experiment.test_langs]
        table[row_name][f"lang-avg"] = aggregate_result_cells(lang_result_cells)  # average of languages
        sd_result_cells = [table[row_name][f"sd-{test_lang}"] for test_lang in experiment.test_langs]
        table[row_name][f"sd-avg"] = aggregate_result_cells(sd_result_cells)  # average of languages sub datasets

    table_df = pd.DataFrame.from_dict(table, orient='index')

    columns = table_df.columns.tolist()
    if not show_sub_datasets_results:
        # remove columns that contain a sub dataset name
        columns = [col for col in columns if not any(sd in col for sd in get_sub_datasets())]

    table_df.to_latex(f"experiment_{experiment.name}.tex", columns=columns)
    table_df.to_html(f"experiment_{experiment.name}.html", columns=columns)
    print(table_df.to_string())


def get_row(experiment, lang_df, sub_datasets=None, metric="f1_macro"):
    """get the results from sub datasets and the test sets of each language"""
    if sub_datasets is None:  # default argument for sub_datasets
        sub_datasets = get_sub_datasets()
    # only consider first entry, the others should have the same structure
    reg_dict = RegexDict(lang_df.iloc[0].summary)
    row_dict = {}
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
                cell = ResultCell(sd_instance_scores.mean(), sd_instance_scores.std(),
                                  sd_instance_scores.min(), sd_instance_scores.max())
                sd_instance_result_cells.append(cell)
            # compute average over all sub dataset classes of a language ==> e.g. de/year
            row_dict[f"sd-{test_lang}-{sub_dataset}"] = aggregate_result_cells(sd_instance_result_cells)
            sd_class_result_cells.append(row_dict[f"sd-{test_lang}-{sub_dataset}"])
        # compute average over all sub dataset languages ==> e.g. de
        row_dict[f"sd-{test_lang}"] = aggregate_result_cells(sd_class_result_cells)

        lang_scores = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/{metric}'])  # series over random seeds
        row_dict[f"lang-{test_lang}"] = ResultCell(lang_scores.mean(), lang_scores.std(),
                                                   lang_scores.min(), lang_scores.max())  # add results per language

    return row_dict


def get_sub_datasets():
    return ['year', 'input_length', 'legal_area', 'origin_region', 'origin_canton']


def aggregate_result_cells(result_cells: List[ResultCell]) -> ResultCell:
    """aggregates a list of result cells into another result cell"""
    result_cells = [rc for rc in result_cells if not rc.empty]  # remove empty result cells
    if len(result_cells) == 0:  # we might get that for en where sub_datasets are missing
        return ResultCell(empty=True)
    means = [rc.mean for rc in result_cells]
    return ResultCell(np.mean(means), np.std(means), np.min(means), np.max(means))


project_name = "SwissJudgmentPredictionCrossLingualTransfer"
# Important overwrite_cache as soon as there are new results
original_df = retrieve_results(project_name, overwrite_cache=False)

create_table(original_df, MonoLingualExperiment())
create_table(original_df, ZeroShotCrossLingualExperiment())
create_table(original_df, MultiLingualExperiment())

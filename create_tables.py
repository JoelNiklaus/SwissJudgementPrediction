from typing import List

import pandas as pd
import numpy as np

from arguments.data_arguments import DataAugmentationType, LegalArea, OriginRegion
from arguments.model_arguments import TrainType, LongInputBertType
from utils.wandb_util import retrieve_results, pd_dp, update_runs

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
    train_sub_datasets = ["None"]
    train_langs = ['de', 'fr', 'it']
    test_langs = ['de', 'fr', 'it']
    sub_dataset_class = None
    show_min = False
    show_lang_aggs = True
    show_sub_dataset_aggs = False
    show_sub_dataset_instance_aggs = False
    show_sub_dataset_lang_aggs = False
    show_sub_dataset_instance_individuals = False
    orient = 'index'


class MonoLingualExperiment(Experiment):
    name = "mono-lingual"
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]


class MultiLingualExperiment(Experiment):
    name = "multi-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    train_langs = ['de,fr,it']


class ZeroShotCrossLingualExperiment(Experiment):
    name = "zero-shot-cross-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    train_langs = ['de,fr', 'de,it', 'fr,it']


class CrossDomainExperiment(Experiment):
    show_min = True
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]
    train_langs = ['de,fr,it']
    show_lang_aggs = False
    show_sub_dataset_instance_aggs = True


class CrossDomainExperimentLegalAreas(CrossDomainExperiment):
    sub_dataset_class = "legal_area"
    name = f"cross-domain-{sub_dataset_class}"
    train_sub_datasets = [legal_area for legal_area in LegalArea]
    train_sub_datasets.append("None")


class CrossDomainExperimentOriginRegions(CrossDomainExperiment):
    sub_dataset_class = "origin_region"
    name = f"cross-domain-{sub_dataset_class}"
    train_sub_datasets = [origin_region for origin_region in OriginRegion]


class ResultCell:
    def __init__(self, mean=0, std=0, min=0, connector='±', show_min=False, empty=False):
        self.mean = mean
        self.std = std
        self.min = min
        self.connector = connector
        self.show_min = show_min
        self.empty = empty  # we got now result
        self.num_decimals = 1

    def round(self, num):
        """Convert a result from 0-1 to a number between 0 and 100 rounded to 2 decimals"""
        return (num * 100).round(self.num_decimals)

    def __str__(self):
        if self.empty:
            return "–"
        mean_std = f"{self.round(self.mean)} {{\small {self.connector} {self.round(self.std)}}}"
        return mean_std + (f" ({self.round(self.min)})" if self.show_min else "")


def get_cols(columns, pattern):
    """Get a subset of the columns matching a given pattern"""
    return [col for col in columns if not re.match(pattern, col)]


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
                cell = ResultCell(sd_instance_scores.mean(), sd_instance_scores.std(), sd_instance_scores.min())
                instance = key.split('/')[-2]
                row_dict[f"sd-{test_lang}-{sub_dataset}-{instance}"] = cell
                sd_instance_result_cells.append(cell)
            # compute average over all sub dataset classes of a language ==> e.g. de/year
            row_dict[f"sd-{test_lang}-{sub_dataset}"] = aggregate_result_cells(sd_instance_result_cells)
            sd_class_result_cells.append(row_dict[f"sd-{test_lang}-{sub_dataset}"])
        # compute average over all sub dataset languages ==> e.g. de
        row_dict[f"sd-{test_lang}"] = aggregate_result_cells(sd_class_result_cells)

        lang_scores = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/{metric}'])  # series over random seeds
        row_dict[f"lang-{test_lang}"] = ResultCell(lang_scores.mean(), lang_scores.std(),
                                                   lang_scores.min())  # add results per language

    # compute average over sd instances over languages
    for sub_dataset in sub_datasets:
        keys = reg_dict.get_matching_keys(f'.+/{sub_dataset}/.+/{metric}')
        instance_result_cells = []
        for key in keys:
            instance = key.split('/')[-2]
            lang_result_cells = []
            for test_lang in experiment.test_langs:
                try:
                    lang_result_cells.append(row_dict[f'sd-{test_lang}-{sub_dataset}-{instance}'])
                # if one of the sub dataset instances was too small for a language so that we did not compute it
                except KeyError:
                    continue  # just ignore it
            row_dict[f'sd-avg-{sub_dataset}-{instance}'] = aggregate_result_cells(lang_result_cells)
            instance_result_cells.append(row_dict[f'sd-avg-{sub_dataset}-{instance}'])
        row_dict[f'sd-avg-{sub_dataset}'] = aggregate_result_cells(instance_result_cells)

    return row_dict


def get_sub_datasets():
    return ['year', 'input_length', 'legal_area', 'origin_region', 'origin_canton']


def aggregate_result_cells(result_cells: List[ResultCell]) -> ResultCell:
    """aggregates a list of result cells into another result cell"""
    result_cells = [rc for rc in result_cells if not rc.empty]  # remove empty result cells
    if len(result_cells) == 0:  # we might get that for en where sub_datasets are missing
        return ResultCell(empty=True)
    means = [rc.mean for rc in result_cells]
    return ResultCell(np.mean(means), np.std(means), np.min(means))


def create_table(df: pd.DataFrame, experiment: Experiment):
    """Creates a table based on the results and the experiment config"""
    table = {}
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

                        for train_sub_dataset in experiment.train_sub_datasets:
                            filter = lambda x: x['data_args']['train_sub_datasets'] == train_sub_dataset
                            tsd_df = da_df[da_df.config.apply(filter)]  # create train_sub_datasets_df
                            if len(tsd_df.index) > 0:
                                # if this fails, there might be some failed/crashed runs which need to be deleted
                                assert len(tsd_df.index) == experiment.num_random_seeds

                                row_name = f"{display_names[model]} " \
                                           f"{display_names[train_type]} " \
                                           f"{display_names[data_augmentation_type]} " \
                                           f"{train_sub_dataset}"
                                # add results per experiment row: merge dicts with same row name (e.g. NativeBERTs)
                                table[row_name] = {**(table[row_name] if row_name in table else {}),
                                                   **(get_row(experiment, tsd_df))}

    # only compute lang-avg and sd-avg across languages here because NativeBERTs are not all available in get_row()
    for row_name in table.keys():
        lang_result_cells = [table[row_name][f"lang-{test_lang}"] for test_lang in experiment.test_langs]
        table[row_name][f"lang-avg"] = aggregate_result_cells(lang_result_cells)  # average of languages
        sd_result_cells = [table[row_name][f"sd-{test_lang}"] for test_lang in experiment.test_langs]
        table[row_name][f"sd-avg"] = aggregate_result_cells(sd_result_cells)  # average of languages sub datasets

    columns = next(iter(table.values())).keys()
    # remove experiment.sub_dataset_class from sub_datasets to be removed
    sub_datasets = [sd for sd in get_sub_datasets() if not sd == experiment.sub_dataset_class]
    # remove columns that contain a sub dataset name
    columns = [col for col in columns if not any(sd in col for sd in sub_datasets)]

    if not experiment.show_lang_aggs:
        # remove columns about languages (on the test set): e.g. lang-fr
        columns = get_cols(columns, "lang-(de|fr|it|en|avg)$")

    if not experiment.show_sub_dataset_aggs:
        # remove columns that contain the sd string but no sub dataset: e.g. sd-de
        columns = get_cols(columns, "sd-(de|fr|it|en|avg)$")

    if not experiment.show_sub_dataset_instance_aggs:
        # remove columns about sub datasets instance averages: e.g. sd-avg-legal_area-social_law
        columns = get_cols(columns, "sd-avg-.+-.+$")

    if not experiment.show_sub_dataset_lang_aggs:
        # remove columns about sub datasets language averages: e.g. sd-de-legal_area
        columns = get_cols(columns, "sd-(de|fr|it|en)-.+$")

    if not experiment.show_sub_dataset_instance_individuals:
        # remove columns about individual sub dataset instances: e.g. sd-de-legal_area-public_law
        columns = get_cols(columns, "sd-(de|fr|it|en)-.+-.+$")

    # set result cell properties
    for row_name, row in table.items():
        for column_name, value in row.items():
            value.show_min = experiment.show_min

    # create pandas dataframe
    table_df = pd.DataFrame.from_dict(table, orient=experiment.orient, columns=columns)

    if experiment.orient == 'index' and experiment.show_sub_dataset_aggs:  # creating multicolumn
        table_df.columns = table_df.columns.str.split('-', expand=True)
        table_df = table_df.sort_index(level=0, axis=1)

        # rename multi col strings for nicer display
        table_df = table_df.rename(columns={"lang": "Test Set", "sd": "Sub Datasets"}, level=0)
    else:
        # rename for nicer display
        rename_dict = {
            "lang-de": "German", "lang-fr": "French", "lang-it": "Italian", "lang-avg": "Average (Languages)",
            "sd-avg-legal_area-public_law": "Public Law", "sd-avg-legal_area-civil_law": "Civil Law",
            "sd-avg-legal_area-penal_law": "Penal Law", "sd-avg-legal_area-social_law": "Social Law",
            "sd-avg-legal_area": "Average (Legal Areas)"
        }
        table_df = table_df.rename(columns=rename_dict)

    table_df.to_latex(f"experiment_{experiment.name}.tex", multicolumn_format="c", escape=False)
    # table_df.to_html(f"experiment_{experiment.name}.html")
    print(table_df.to_string())


project_name = "SwissJudgmentPredictionCrossLingualTransfer"
# Important overwrite_cache as soon as there are new results
original_df = retrieve_results(project_name, overwrite_cache=False)

# create_table(original_df, MonoLingualExperiment())
# create_table(original_df, MultiLingualExperiment())
# create_table(original_df, ZeroShotCrossLingualExperiment())

create_table(original_df, CrossDomainExperimentLegalAreas())
# create_table(original_df, CrossDomainExperimentOriginRegions())

from typing import List

import pandas as pd
import numpy as np

from arguments.data_arguments import DataAugmentationType, LegalArea, OriginRegion, SubDataset, Jurisdiction
from arguments.model_arguments import TrainType
from evaluation.experiments import (Experiment,
                                    MonoLingualExperiment,
                                    MultiLingualExperiment,
                                    ZeroShotCrossLingualExperiment,
                                    CrossDomainLegalAreasExperiment,
                                    CrossDomainOriginRegionsExperiment,
                                    CrossJurisdictionExperiment,
                                    CrossJurisdictionLegalAreasExperiment)
from evaluation.result_cell import ResultCell
from utils.wandb_util import retrieve_results

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


def isNativeBert(model_name):
    return display_names[model_name] == "NativeBERT"


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
    LegalArea.PUBLIC_LAW: "Public Law",
    LegalArea.CIVIL_LAW: "Civil Law",
    LegalArea.PENAL_LAW: "Penal Law",
    LegalArea.SOCIAL_LAW: "Social Law",
    OriginRegion.ZURICH: "Zürich",
    OriginRegion.EASTERN_SWITZERLAND: "Eastern Switzerland",
    OriginRegion.CENTRAL_SWITZERLAND: "Central Switzerland",
    OriginRegion.NORTHWESTERN_SWITZERLAND: "Northwestern Switzerland",
    OriginRegion.ESPACE_MITTELLAND: "Espace Mittelland",
    OriginRegion.REGION_LEMANIQUE: "Région lémanique",
    OriginRegion.TICINO: "Ticino",
    OriginRegion.FEDERATION: "Federation",
    'None': 'All',
    Jurisdiction.SWITZERLAND: "CH",
    Jurisdiction.INDIA: "IN",
    Jurisdiction.BOTH: "CH+IN",
}


def get_cols(columns, pattern):
    """Get a subset of the columns matching a given pattern"""
    return [col for col in columns if not re.match(pattern, col)]


def get_row(experiment, lang_df, sub_datasets=None):
    """get the results from sub datasets and the test sets of each language"""
    if sub_datasets is None:  # default argument for sub_datasets
        sub_datasets = get_sub_datasets()
    # only consider first entry, the others should have the same structure
    reg_dict = RegexDict(lang_df.iloc[0].summary)
    row_dict = {}
    for test_lang in experiment.test_langs:
        matches = reg_dict.get_matching_keys(f'test/{test_lang}/{experiment.metric}')
        if next(matches, -1) == -1:  # if we did not find any of entries in that language
            continue  # skip it
        sd_class_result_cells = []
        # we compute it for every sub dataset class so that we can get the average over all sub datasets later
        for sub_dataset_class in sub_datasets:
            sub_dataset = sub_dataset_class.get_dataset_column_name()
            keys = reg_dict.get_matching_keys(f'{test_lang}/{sub_dataset}/.+/{experiment.metric}')
            sd_instance_result_cells = []
            for key in keys:
                try:
                    # compute average over all instances of a sub dataset ==> e.g. year: 2017, .., 2020
                    sd_instance_scores = lang_df.summary.apply(lambda x: x[key])  # series over random seeds
                    support_key = key.replace(experiment.metric, "support")
                    sd_instance_supports = lang_df.summary.apply(lambda x: x[support_key])
                except KeyError:
                    continue  # probably we wanted to retrieve something like 'de/origin_region/Region_Lemanique/f1_macro' (there are no German cases in Region_Lemanique, therefore we don't have any f1_macro)
                # the number of samples should always be the same
                assert int(sd_instance_supports.mean()) == sd_instance_supports.iloc[0]
                cell = ResultCell(sd_instance_scores.mean(), sd_instance_scores.std(), sd_instance_scores.min(),
                                  sd_instance_supports.iloc[0])
                instance = key.split('/')[-2]
                if instance == 'Région lémanique':
                    instance = 'Region_Lemanique'  # Dirty hack to rename some old unfortunate spelling
                row_dict[f"sd-{test_lang}-{sub_dataset}-{instance}"] = cell
                sd_instance_result_cells.append(cell)
            # compute average over all sub dataset classes of a language ==> e.g. de/year
            row_dict[f"sd-{test_lang}-{sub_dataset}"] = aggregate_result_cells(sd_instance_result_cells)
            sd_class_result_cells.append(row_dict[f"sd-{test_lang}-{sub_dataset}"])
        # compute average over all sub dataset languages ==> e.g. de
        row_dict[f"sd-{test_lang}"] = aggregate_result_cells(sd_class_result_cells)

        # series over random seeds
        lang_scores = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/{experiment.metric}'])
        lang_supports = lang_df.summary.apply(lambda x: x[f'test/{test_lang}/samples'])
        # the number of samples should always be the same
        assert int(lang_supports.mean()) == lang_supports.iloc[0]
        row_dict[f"lang-{test_lang}"] = ResultCell(lang_scores.mean(), lang_scores.std(),
                                                   lang_scores.min(), lang_supports.iloc[0])  # add results per language

    return row_dict


def get_sub_datasets():
    return SubDataset.__subclasses__()


def aggregate_result_cells(result_cells: List[ResultCell], use_weighted_average=False) -> ResultCell:
    """aggregates a list of result cells into another result cell"""
    result_cells = [rc for rc in result_cells if not rc.empty]  # remove empty result cells
    if len(result_cells) == 0:  # we might get that for en where sub_datasets are missing
        return ResultCell(empty=True)
    means = [rc.mean for rc in result_cells]
    mean = np.mean(means)
    if use_weighted_average:
        supports = [rc.support for rc in result_cells]
        assert all(support for support in supports)  # we need a support for that
        mean = np.average(means, weights=supports)  # overwrite mean with weighted average
    return ResultCell(mean, np.std(means), np.min(means))


def compute_averages(experiment, table):
    """
    Compute averages over languages and sub datasets not available in the get_row function
    since they come from different models (NativeBERTs)
    """
    for row_name in table.keys():
        if experiment.show_lang_aggs:
            lang_result_cells = [table[row_name][f"lang-{test_lang}"] for test_lang in experiment.test_langs]
            table[row_name][f"lang-avg"] = aggregate_result_cells(lang_result_cells)  # average of languages
        if experiment.show_sub_dataset_aggs:
            sd_result_cells = [table[row_name][f"sd-{test_lang}"] for test_lang in experiment.test_langs]
            table[row_name][f"sd-avg"] = aggregate_result_cells(sd_result_cells)  # average of languages sub datasets

        if experiment.show_sub_dataset_instance_aggs:
            # compute average over sd instances over languages
            for sub_dataset_class in get_sub_datasets():
                sub_dataset = sub_dataset_class.get_dataset_column_name()
                instances = [instance.value for instance in sub_dataset_class]
                instance_result_cells = []
                for instance in instances:
                    lang_result_cells = []
                    for test_lang in experiment.test_langs:
                        try:
                            lang_result_cells.append(table[row_name][f'sd-{test_lang}-{sub_dataset}-{instance}'])
                        # if one of the sub dataset instances was too small for a language so that we did not compute it
                        except KeyError:
                            continue  # just ignore it
                    result_cell = aggregate_result_cells(lang_result_cells, experiment.use_support_weighted_average)
                    table[row_name][f'sd-avg-{sub_dataset}-{instance}'] = result_cell
                    instance_result_cells.append(table[row_name][f'sd-avg-{sub_dataset}-{instance}'])
                table[row_name][f'sd-avg-{sub_dataset}'] = aggregate_result_cells(instance_result_cells)
    return table


def get_columns_for_display(experiment, table):
    """Compiles the columns that should be shown in the final output table"""
    columns = next(iter(table.values())).keys()
    # remove experiment.sub_dataset_class from sub_datasets to be removed
    sub_datasets = [sd for sd in get_sub_datasets() if not sd == experiment.sub_dataset_class]
    # remove columns that contain a sub dataset name
    columns = [col for col in columns if not any(sd.get_dataset_column_name() in col for sd in sub_datasets)]
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
    return columns


def fill_table(df, experiment):
    """Fills the table with the individual results"""
    # TODO maybe add another config param for experiment name for easy filtering
    table = {}
    for train_lang in experiment.train_langs:
        for train_type in experiment.train_types:
            for model_type in experiment.model_types:
                for model in get_bert_models(train_lang):
                    run_name = f'{train_type}-{model}-{model_type}-{train_lang}-'
                    lang_df = df[df.name.str.contains(run_name)]  # filter by run_name

                    for data_augmentation_type in experiment.data_augmentation_types:
                        filter = lambda x: x['data_args']['data_augmentation_type'] == data_augmentation_type
                        da_df = lang_df[lang_df.config.apply(filter)]  # create data_augmentation_df

                        for train_sub_dataset in experiment.train_sub_datasets:
                            filter = lambda x: x['data_args']['train_sub_datasets'] == train_sub_dataset
                            tsd_df = da_df[da_df.config.apply(filter)]  # create train_sub_datasets_df

                            for jurisdiction in experiment.jurisdictions:
                                filter = lambda x: x['data_args']['jurisdiction'] == jurisdiction
                                j_df = tsd_df[tsd_df.config.apply(filter)]  # create jurisdiction_df

                                if len(j_df.index) > 0:
                                    # if this fails, there might be some failed/crashed runs which need to be deleted
                                    assert len(j_df.index) == experiment.num_random_seeds

                                    row_name = f"{display_names[model]} " \
                                               f"{display_names[train_type]} "
                                    # if only trained on one language call it "mono", if trained on several call it "multi"
                                    row_name += f"mono " if len(train_lang) == 2 else "multi "
                                    row_name += f"{display_names[data_augmentation_type]} " if len(
                                        experiment.data_augmentation_types) > 1 else ""
                                    row_name += f"{display_names[train_sub_dataset]} " if len(
                                        experiment.train_sub_datasets) > 1 else ""
                                    row_name += f"{display_names[jurisdiction]} " if len(
                                        experiment.jurisdictions) > 1 else ""
                                    # add results per experiment row: merge dicts with same row name (e.g. NativeBERTs)
                                    table[row_name] = {**(table[row_name] if row_name in table else {}),
                                                       **(get_row(experiment, j_df))}
    return table


def create_table(df: pd.DataFrame, experiment: Experiment):
    """Creates a table based on the results and the experiment config"""
    table = fill_table(df, experiment)

    # only compute lang-avg and sd-avg across languages here because NativeBERTs are not all available in get_row()
    table = compute_averages(experiment, table)

    columns = get_columns_for_display(experiment, table)

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
            "lang-de": "German", "lang-fr": "French", "lang-it": "Italian", "lang-avg": "Average",
            "sd-avg-legal_area-public_law": "Public Law", "sd-avg-legal_area-civil_law": "Civil Law",
            "sd-avg-legal_area-penal_law": "Penal Law", "sd-avg-legal_area-social_law": "Social Law",
            "sd-avg-legal_area": "Average",
            "sd-avg-origin_region-Espace_Mittelland": "Espace Mittelland", "sd-avg-origin_region-Zürich": "Zürich",
            "sd-avg-origin_region-Region_Lemanique": "Région lémanique",
            "sd-avg-origin_region-Federation": "Federation",
            "sd-avg-origin_region-Ticino": "Ticino", "sd-avg-origin_region-Central_Switzerland": "Central Switzerland",
            "sd-avg-origin_region-Eastern_Switzerland": "Eastern Switzerland",
            "sd-avg-origin_region-Northwestern_Switzerland": "Northwestern Switzerland",
            "sd-avg-origin_region": "Average",
        }
        table_df = table_df.rename(columns=rename_dict)

    print(experiment.name)
    if experiment.save_to_latex:
        table_df.to_latex(f"experiment_{experiment.name}.tex", multicolumn_format="c", escape=False)
    if experiment.save_to_html:
        table_df.to_html(f"experiment_{experiment.name}.html")
    print(table_df.to_string())


if __name__ == '__main__':
    project_name = "SwissJudgmentPredictionCrossLingualTransfer"
    # Important overwrite_cache as soon as there are new results
    original_df = retrieve_results(project_name, overwrite_cache=False)

    create_table(original_df, MonoLingualExperiment())
    create_table(original_df, MultiLingualExperiment())
    create_table(original_df, ZeroShotCrossLingualExperiment())

    create_table(original_df, CrossDomainLegalAreasExperiment())
    create_table(original_df, CrossDomainOriginRegionsExperiment())

    create_table(original_df, CrossJurisdictionExperiment())
    create_table(original_df, CrossJurisdictionLegalAreasExperiment())

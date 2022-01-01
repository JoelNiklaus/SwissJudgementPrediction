from arguments.data_arguments import DataAugmentationType, LegalArea, OriginRegion
from arguments.model_arguments import LongInputBertType, TrainType


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
    save_to_latex = True
    save_to_html = False


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
    train_langs = ['de,fr,it', 'de', 'fr', 'it']
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
    train_sub_datasets.append("None")

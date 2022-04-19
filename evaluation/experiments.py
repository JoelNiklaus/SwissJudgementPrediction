from arguments.data_arguments import DataAugmentationType, LegalArea, OriginRegion, Jurisdiction
from arguments.model_arguments import LongInputBertType, TrainType


class Experiment:
    name = "experiment"
    num_random_seeds = 3
    model_types = [LongInputBertType.HIERARCHICAL]
    train_types = [TrainType.FINETUNE]
    data_augmentation_types = [DataAugmentationType.NO_AUGMENTATION]
    jurisdictions = [Jurisdiction.SWITZERLAND]
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
    metric = "f1_macro"
    # whether or not to use support to weight the averages (if False: each language has equal weight)
    use_support_weighted_average = False


class MonoLingualExperiment(Experiment):
    name = "mono-lingual"
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]


class MultiLingualExperiment(Experiment):
    name = "multi-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]
    train_langs = ['de,fr,it']


class ZeroShotCrossLingualExperiment(Experiment):
    name = "zero-shot-cross-lingual"
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    train_langs = ['de,fr', 'de,it', 'fr,it']


class CrossDomainExperiment(Experiment):
    train_types = [TrainType.FINETUNE, TrainType.ADAPTERS]
    data_augmentation_types = [DataAugmentationType.TRANSLATION, DataAugmentationType.NO_AUGMENTATION]
    train_langs = ['de,fr,it', 'de', 'fr', 'it']
    show_lang_aggs = False
    show_sub_dataset_instance_aggs = True
    use_support_weighted_average = True
    show_min = False


class CrossDomainLegalAreasExperiment(CrossDomainExperiment):
    sub_dataset_class = LegalArea
    name = f"cross-domain-{sub_dataset_class.get_dataset_column_name()}"
    train_sub_datasets = [legal_area for legal_area in LegalArea]
    train_sub_datasets.append("None")


class CrossDomainOriginRegionsExperiment(CrossDomainExperiment):
    sub_dataset_class = OriginRegion
    name = f"cross-domain-{sub_dataset_class.get_dataset_column_name()}"
    train_sub_datasets = [origin_region for origin_region in OriginRegion]
    train_sub_datasets.append("None")


class CrossJurisdictionExperiment(Experiment):
    name = "cross-jurisdiction"
    data_augmentation_types = [DataAugmentationType.TRANSLATION]
    jurisdictions = [Jurisdiction.SWITZERLAND, Jurisdiction.INDIA, Jurisdiction.BOTH]
    train_langs = ['de,fr,it']
    test_langs = ['de', 'fr', 'it', 'en']


class CrossJurisdictionLegalAreasExperiment(CrossDomainLegalAreasExperiment, CrossJurisdictionExperiment):
    sub_dataset_class = LegalArea
    name = f"cross-jurisdiction-cross-domain-{sub_dataset_class.get_dataset_column_name()}"
    data_augmentation_types = [DataAugmentationType.TRANSLATION]
    jurisdictions = [Jurisdiction.SWITZERLAND, Jurisdiction.INDIA, Jurisdiction.BOTH]
    train_langs = ['de,fr,it']
    test_langs = ['de', 'fr', 'it', 'en']
    train_sub_datasets = ["None"]

from transformers.file_utils import ExplicitEnum
from typing import Optional

from dataclasses import dataclass, field


class ProblemType(str, ExplicitEnum):
    REGRESSION = "regression"
    SINGLE_LABEL_CLASSIFICATION = "single_label_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"


class SegmentationType(str, ExplicitEnum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    BLOCK = "block"
    OVERLAPPING = "overlapping"


class DataAugmentationType(str, ExplicitEnum):
    TRANSLATION = "translation"
    BACK_TRANSLATION = "back_translation"
    NO_AUGMENTATION = "no_augmentation"


class SubDataset(ExplicitEnum):
    """Only legal area and origin canton make sense for training"""

    # INPUT_LENGTH = "input_length"
    # YEAR = "year"
    # LEGAL_AREA = "legal_area"
    # ORIGIN_REGION = "origin_region"
    # ORIGIN_CANTON = "origin_canton"
    # ORIGIN_COURT = "origin_court"
    # ORIGIN_CHAMBER = "origin_chamber"
    @staticmethod
    def from_str(label):
        try:
            return LegalArea(label)
        except ValueError:
            try:
                return OriginRegion(label)
            except ValueError:
                try:
                    return OriginCanton(label)
                except ValueError:
                    message = f"Your label {label} is neither a LegalArea nor an OriginRegion nor an OriginCanton"
                    raise ValueError(message)


class LegalArea(str, SubDataset):
    PUBLIC_LAW = "public_law"
    CIVIL_LAW = "civil_law"
    PENAL_LAW = "penal_law"
    SOCIAL_LAW = "social_law"

    # INSURANCE_LAW = "insurance_law"  # there is no evaluation set
    # OTHER = "other"  # cannot be used for training: too small

    @classmethod
    def get_dataset_column_name(cls):
        return "legal_area"


class OriginRegion(str, SubDataset):
    ZURICH = "Zürich"
    EASTERN_SWITZERLAND = "Eastern_Switzerland"
    CENTRAL_SWITZERLAND = "Central_Switzerland"
    NORTHWESTERN_SWITZERLAND = "Northwestern_Switzerland"
    ESPACE_MITTELLAND = "Espace_Mittelland"
    REGION_LEMANIQUE = "Region_Lemanique"
    TICINO = "Ticino"
    FEDERATION = "Federation"

    @classmethod
    def get_dataset_column_name(cls):
        return "origin_region"


class OriginCanton(str, SubDataset):
    ZURICH = "ZH"
    BERNE = "BE"
    LUCERNE = "LU"
    URI = "UR"
    SCHWYZ = "SZ"
    OBWALDEN = "OW"
    NIDWALDEN = "NW"
    GLARUS = "GL"
    ZUG = "ZG"
    FRIBOURG = "FR"
    SOLEURE = "SO"
    BASEL_CITY = "BS"
    BASEL_COUNTRY = "BL"
    SHAFFHAUSEN = "SH"
    APPENZELL_OUTER_RHODES = "AR"
    APPENZELL_INNER_RHODES = "AI"
    ST_GALL = "SG"
    GRISONS = "GR"
    ARGOVIA = "AG"
    THURGOVIA = "TG"
    TICINO = "TI"
    VAUD = "VD"
    VALAIS = "VS"
    NEUCHATEL = "NE"
    GENEVE = "GE"
    JURA = "JU"
    CONFEDERATION = "CH"

    @classmethod
    def get_dataset_column_name(cls):
        return "origin_canton"


class Jurisdiction(str, ExplicitEnum):
    SWITZERLAND = "switzerland"
    INDIA = "india"
    BOTH = "both"


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments
    to be able to specify them on the command line.
    """
    experiment_name: Optional[str] = field(
        default="",  # e.g. cross-domain, multi-lingual, mono-lingual
        metadata={"help": "The name of the experiment this run is part of."}
    )
    tune_hyperparams: bool = field(
        default=False, metadata={"help": "Whether or not to tune the hyperparameters before training."},
    )
    max_seq_len: Optional[int] = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_segments: Optional[int] = field(
        default=8,
        metadata={
            "help": "The maximum number of segments (paragraphs/sentences) to be considered. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seg_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum segment (paragraph/sentences) length to be considered. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_seg_len: Optional[int] = field(
        default=64,
        metadata={
            "help": "If a segment is smaller than this many characters, it will be concatenated with the next segment."
        },
    )
    segmentation_type: SegmentationType = field(
        default=SegmentationType.BLOCK,
        metadata={
            "help": "How to split the text into segments when using hierarchical BERT."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_len`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
                    "Padding to 'longest' may lead to problems in hierarchical and long bert."
        },
    )
    data_augmentation_type: DataAugmentationType = field(
        default=DataAugmentationType.NO_AUGMENTATION, metadata={"help": "What type of data augmentation to use"},
    )
    jurisdiction: Jurisdiction = field(
        default=Jurisdiction.SWITZERLAND, metadata={"help": "Which jurisdiction to include data from"},
    )
    train_sub_datasets: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether to train on certain sub datasets only or not (training on all of them)."
        },
    )
    test_on_sub_datasets: bool = field(
        default=False,
        metadata={
            "help": "Whether to test on the sub datasets or not."
        },
    )
    log_all_predictions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log the individual predictions on the train set and eval set as well."
        },
    )
    problem_type: ProblemType = field(
        default=ProblemType.SINGLE_LABEL_CLASSIFICATION,
        metadata={
            "help": "Problem type for XxxForSequenceClassification models. "
        },
    )
    task_name: Optional[str] = field(
        default="sjp",  # SwissJudgementPrediction
        metadata={"help": "The name of the task to train on."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})

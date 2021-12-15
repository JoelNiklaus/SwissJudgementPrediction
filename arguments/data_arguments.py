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


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments
    to be able to specify them on the command line.
    """
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
        default=True,  # TODO change to false again if everything works
        metadata={
            "help": "Whether to pad all samples to `max_seq_len`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
                    "Padding to 'longest' may lead to problems in hierarchical and long bert."
        },
    )
    data_augmentation_type: DataAugmentationType = field(
        default=DataAugmentationType.NO_AUGMENTATION, metadata={"help": "What type of data augmentation to use"},
    )
    test_on_sub_datasets: bool = field(
        default=False,
        metadata={
            "help": "Whether to test on the sub datasets or not."
        },
    )
    problem_type: ProblemType = field(
        default="single_label_classification",
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

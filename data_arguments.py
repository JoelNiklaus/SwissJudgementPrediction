from typing import Optional

from dataclasses import dataclass, field


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
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,  # TODO change to false again if everything works
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
                    "Padding to 'longest' may lead to problems in hierarchical and long bert."
        },
    )
    test_on_sub_datasets: bool = field(
        default=False,
        metadata={
            "help": "Whether to test on the sub datasets or not."
        },
    )
    problem_type: str = field(
        default="single_label_classification",
        metadata={
            "help": "Problem type for XxxForSequenceClassification models. "
                    "Can be one of (\"regression\", \"single_label_classification\", \"multi_label_classification\")."
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

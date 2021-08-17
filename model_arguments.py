from typing import Optional

from dataclasses import dataclass, field

long_input_bert_types = ['long', 'longformer', 'hierarchical']


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    long_input_bert_type: str = field(
        default=None,
        metadata={"help": f"Which bert type to use for handling long text inputs. "
                          f"Currently the following types are supported: {long_input_bert_types}."},
    )
    use_adapters: bool = field(
        default=False,
        metadata={
            "help": "If True uses adapters instead of finetuning the model."
        },
    )
    evaluation_language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None. "
                                        "Can also be set to 'all'"},
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    label_imbalance_method: Optional[str] = field(
        default=None,
        metadata={"help": "Whether or not to use any method to combat label imbalance. "
                          "Available are 'class_weights', 'oversampling' and 'undersampling'."
                          "'class_weights' applies a special term to the loss function which gives more weight to the minority class. "
                          "(Because this messes with the loss function, label smoothing does not work if this is enabled) "
                          "'oversampling' oversamples the minority class to match the number of samples in the majority class. "
                          "'undersampling' undersamples the majority class to match the number of samples in the minority class."
                  },
    )
    prediction_threshold: int = field(
        default=0,
        metadata={
            "help": "Used in multilabel classification for determining when a given label is assigned. "
                    "This is normally 0 when using the tanh function in the output layer "
                    "and 0.5 if the sigmoid function is used."
                    "This is a hyperparameter which can additionally be tuned to improve the "
                    "multilabel classification performance as discussed here: "
                    "https://www.csie.ntu.edu.tw/~cjlin/papers/threshold.pdf"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

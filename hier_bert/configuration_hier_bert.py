from typing import Optional

from transformers import BertConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HierBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        max_segments (:obj:`int`, `optional`, defaults to 4):
            The maximum number of segments (paragraphs/sentences) to be considered. Sequences longer
            than this will be truncated, sequences shorter will be padded.
        max_segment_length (:obj:`int`, `optional`, defaults to 512):
            The maximum segment (paragraph/sentences) length to be considered. Sequences longer
            than this will be truncated, sequences shorter will be padded.
        segment_encoder_type (:obj:`str`, `optional`, defaults to "transformer"):
            Whether to use a transformer or an lstm to encode the segment level representations.
    """

    model_type = "bert"

    def __init__(
            self,
            max_segments: Optional[int] = 4,
            max_segment_length: Optional[int] = 512,
            segment_encoder_type: Optional[str] = "transformer",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.segment_encoder_type = segment_encoder_type

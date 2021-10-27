from transformers.utils import logging

from hier_roberta.configuration_hier_roberta import HierRobertaConfig

logger = logging.get_logger(__name__)


class HierCamembertConfig(HierRobertaConfig):
    r"""
    This class overrides :class:`~transformers.HierRobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = "camembert"

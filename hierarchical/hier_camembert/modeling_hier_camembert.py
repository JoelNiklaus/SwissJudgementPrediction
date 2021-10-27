from transformers import add_start_docstrings
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_START_DOCSTRING

from hierarchical.hier_camembert.configuration_hier_camembert import HierCamembertConfig
from hierarchical.hier_roberta.modeling_hier_roberta import HierRobertaForSequenceClassification


@add_start_docstrings(
    """
    Hierarchical Camembert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class HierCamembertForSequenceClassification(HierRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.HierRobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = HierCamembertConfig

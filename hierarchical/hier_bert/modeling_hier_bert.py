import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel, ModelWithHeadsAdaptersMixin, add_start_docstrings
from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, \
    _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC


@add_start_docstrings(
    """
    Hierarchical Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class HierBertForSequenceClassification(ModelWithHeadsAdaptersMixin, BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.max_segments = config.max_segments
        self.max_segment_length = config.max_segment_length
        self.segment_encoder_type = config.segment_encoder_type
        self.config = config

        self.bert = BertModel(config)

        def sinusoidal_init(num_embeddings: int, embedding_dim: int):
            # keep dim 0 for padding token position encoding zero vector
            position_enc = np.array([
                [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
                if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

            position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
            position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
            return torch.from_numpy(position_enc).type(torch.FloatTensor)

        if self.segment_encoder_type == "lstm":
            # Init segment-wise BiLSTM-based encoder
            self.segment_encoder = nn.LSTM(self.hidden_size, self.hidden_size,
                                           bidirectional=True, num_layers=1, batch_first=True)
            self.down_project = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        elif self.segment_encoder_type == "transformer":
            # Init sinusoidal positional embeddings
            weight = sinusoidal_init(self.max_segments + 1, self.hidden_size)
            self.seg_pos_embeddings = nn.Embedding(self.max_segments + 1, self.hidden_size,
                                                   padding_idx=0, _weight=weight)

            # Init segment-wise transformer-based encoder
            self.segment_encoder = nn.Transformer(d_model=self.hidden_size,
                                                  nhead=config.num_attention_heads,
                                                  batch_first=True, dim_feedforward=config.intermediate_size,
                                                  activation=config.hidden_act,
                                                  dropout=config.hidden_dropout_prob,
                                                  layer_norm_eps=config.layer_norm_eps,
                                                  num_encoder_layers=2, num_decoder_layers=0).encoder

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Input (samples, segments, max_segment_length) --> (16, 10, 510)
        # Squash samples and segments into a single axis (samples * segments, max_segment_length) --> (160, 512)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))

        outputs = self.bert(
            input_ids_reshape,
            attention_mask=attention_mask_reshape,
            token_type_ids=token_type_ids_reshape,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
        )

        # TODO the outputs might have the wrong shape here
        # in the original model it takes outputs[1]

        encoder_outputs = outputs[0].contiguous().view(input_ids.size(0), self.max_segments,
                                                       self.max_segment_length, self.hidden_size)

        # Gather CLS per segment --> (16, 10, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        if self.segment_encoder_type == 'lstm':
            # LSTMs on top of segment encodings --> (16, 10, 1536)
            lstms = self.segment_encoder(encoder_outputs)

            # Reshape LSTM outputs to split directions -->  (16, 10, 2, 768)
            reshaped_lstms = lstms[0].view(input_ids.size(0), self.max_segments, 2, self.hidden_size)

            # Concatenate of first and last hidden states -->  (16, 1536)
            seg_encoder_outputs = torch.cat((reshaped_lstms[:, -1, 0, :], reshaped_lstms[:, 0, 1, :]), -1)

            # Down-project -->  (16, 768)
            seg_encoder_outputs = self.down_project(seg_encoder_outputs)

        if self.segment_encoder_type == 'transformer':
            # Transformer on top of segment encodings --> (16, 10, 768)
            # Infer real segments, i.e., mask paddings (like attention_mask but on a segment level)
            seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
            # Infer and collect segment positional embeddings
            seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
            # Add segment positional embeddings to segment inputs
            encoder_outputs += self.seg_pos_embeddings(seg_positions)

            # Encode segments with segment-wise transformer
            seg_encoder_outputs = self.segment_encoder(encoder_outputs)

            # Collect document representation
            seg_encoder_outputs, _ = torch.max(seg_encoder_outputs, 1)

        pooled_output = self.dropout(seg_encoder_outputs)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

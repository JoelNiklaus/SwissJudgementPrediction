from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.file_utils import ModelOutput


# TODO why not use transformers.modeling_outputs.SequenceClassifierOutput here?
@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# TODO subclass BertModel, BertConfig and BertTokenizer to make it more clean and to override save_pretrained() so that the seg_encoder is saved too
class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments, max_segment_length, cls_token_id, sep_token_id, device,
                 seg_encoder_type="lstm"):
        super(HierarchicalBert, self).__init__()
        supported_models = ['bert', 'camembert', 'xlm-roberta']
        assert encoder.config.model_type in supported_models  # other models are not supported so far
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.device = device
        self.seg_encoder_type = seg_encoder_type
        if self.seg_encoder_type == "lstm":
            self.seg_encoder = nn.LSTM(encoder.config.hidden_size, encoder.config.hidden_size,
                                       bidirectional=True, num_layers=1, batch_first=True)
        if self.seg_encoder_type == "transformer":
            # TODO make this work too
            self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size, num_encoder_layers=2).encoder
        if self.seg_encoder_type == "linear":
            # TODO make this work too
            self.seg_encoder = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.down_project = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)

    def forward(self,
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
        # Hypothetical example (samples, max_document_size) --> (16, 5110) and max_segments == 10
        # Reshape samples into segments # (samples, segments, max_segment_length) --> (16, 10, 510)
        input_ids = torch.reshape(input_ids, (input_ids.size(0), self.max_segments, self.max_segment_length - 2))
        attention_mask = torch.reshape(attention_mask,
                                       (attention_mask.size(0), self.max_segments, self.max_segment_length - 2))
        token_type_ids = torch.reshape(token_type_ids,
                                       (token_type_ids.size(0), self.max_segments, self.max_segment_length - 2))

        # Put back [CLS] and [SEP] tokens per segment (16, 10, 510) --> (16, 10, 512)
        cls_input_ids = torch.ones((input_ids.size(0), self.max_segments, 1), dtype=torch.int,
                                   device=self.device) * self.cls_token_id
        sep_input_ids = torch.ones((input_ids.size(0), self.max_segments, 1), dtype=torch.int,
                                   device=self.device) * self.sep_token_id
        extra_attention_masks = torch.ones((attention_mask.size(0), self.max_segments, 1), dtype=torch.int,
                                           device=self.device)
        extra_token_type_ids = torch.zeros((token_type_ids.size(0), self.max_segments, 1), dtype=torch.int,
                                           device=self.device)
        input_ids = torch.cat([cls_input_ids, input_ids, sep_input_ids], 2)
        attention_mask = torch.cat([extra_attention_masks, attention_mask, extra_attention_masks], 2)
        token_type_ids = torch.cat([extra_token_type_ids, token_type_ids, extra_token_type_ids], 2)

        # Squash samples and segments into a single axis (samples * segments, max_segment_length) --> (160, 512)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))

        # Encode segments with BERT --> (160, 512, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (samples, segments, max_segment_length, output_size) --> (16, 10, 512, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS per segment --> (16, 10, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        if self.seg_encoder_type == 'lstm':
            # LSTMs on top of segment encodings --> (16, 10, 1536)
            lstms = self.seg_encoder(encoder_outputs)

            # Reshape LSTM outputs to split directions -->  (16, 10, 2, 768)
            reshaped_lstms = lstms[0].view(input_ids.size(0), self.max_segments, 2, self.hidden_size)

            # Concatenate of first and last hidden states -->  (16, 1536)
            seg_encoder_outputs = torch.cat((reshaped_lstms[:, -1, 0, :], reshaped_lstms[:, 0, 1, :]), -1)

        if self.seg_encoder_type == 'transformer':
            # Transformer on top of segment encodings --> (16, 10, 768)
            transformer = self.seg_encoder(encoder_outputs)
            # TODO make this work too

        if self.seg_encoder_type == 'linear':
            # Transformer on top of segment encodings --> (16, 10, 768)
            transformer = self.seg_encoder(encoder_outputs)
            # TODO make this work too

        # Down-project -->  (16, 768)
        outputs = self.down_project(seg_encoder_outputs)
        return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)

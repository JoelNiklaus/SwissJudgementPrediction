import faulthandler

import torch
from transformers import (
    AutoTokenizer, AutoConfig,
)
from transformers.adapters.configuration import AdapterConfig

from hierarchical.hier_bert.configuration_hier_bert import HierBertConfig
from hierarchical.hier_bert.modeling_hier_bert import HierBertForSequenceClassification

faulthandler.enable()

padding = "max_length"
max_segments = 4
max_seg_len = 512
task_name = "sjp"
device = 'cpu'  # 'cpu'  # 'cuda:0'

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config_dict = config.to_dict()
config = HierBertConfig(max_segments=max_segments, max_segment_length=max_seg_len, segment_encoder_type="transformer", **config_dict)
model = HierBertForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

adapter_config = AdapterConfig.load("pfeiffer", non_linearity=None, reduction_factor=None)
model.add_adapter(task_name, config=adapter_config)
model.train_adapter([task_name])
model.set_active_adapters(task_name)

def preprocess_function(batch):
    batch['segments'] = []
    tokenized = tokenizer(batch["text"], padding=padding, truncation=True,
                          max_length=max_segments * max_seg_len)
    for ids in tokenized['input_ids']:
        # convert ids to tokens and then back to strings
        id_blocks = [ids[i:i + max_seg_len] for i in range(0, len(ids), max_seg_len)]
        token_blocks = [tokenizer.convert_ids_to_tokens(ids) for ids in id_blocks]
        string_blocks = [tokenizer.convert_tokens_to_string(tokens) for tokens in token_blocks]
        batch['segments'].append(string_blocks)

    # Tokenize the texts
    def append_zero_segments(case_encodings):
        """appends a list of zero segments to the encodings to make up for missing segments"""
        return case_encodings + [[0] * max_seg_len] * (max_segments - len(case_encodings))

    tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    for case in batch['segments']:
        case_encodings = tokenizer(case[:max_segments], padding=padding, truncation=True,
                                   max_length=max_seg_len, return_token_type_ids=True)
        tokenized['input_ids'].append(append_zero_segments(case_encodings['input_ids']))
        tokenized['attention_mask'].append(append_zero_segments(case_encodings['attention_mask']))
        tokenized['token_type_ids'].append(append_zero_segments(case_encodings['token_type_ids']))
    del batch['segments']

    return tokenized


batch = {"text": ['a' * max_segments * max_seg_len] * 4}
batch = preprocess_function(batch)
print("tokenization successful")

outputs = model(input_ids=torch.tensor(batch['input_ids'], device=device),
                attention_mask=torch.tensor(batch['attention_mask'], device=device),
                token_type_ids=torch.tensor(batch['token_type_ids'], device=device))
print(outputs)

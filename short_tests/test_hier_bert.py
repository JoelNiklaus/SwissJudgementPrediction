import faulthandler
import random

import torch
from transformers import (
    AutoTokenizer, AutoConfig,
)
from transformers.adapters.configuration import AdapterConfig

from hierarchical.hier_xlm_roberta.configuration_hier_xlm_roberta import HierRobertaConfig
from hierarchical.hier_xlm_roberta.modeling_hier_xlm_roberta import HierXLMRobertaForSequenceClassification
from transformers import AutoModel
faulthandler.enable()

padding = "max_length"
max_segments = 4
max_seg_len = 64
task_name = "sjp"
device = 'cpu'  # 'cpu'  # 'cuda:0'

model_name = "xlm-roberta-base"
config = AutoConfig.from_pretrained(model_name)
config_dict = config.to_dict()
config = HierRobertaConfig(max_segments=max_segments, max_segment_length=max_seg_len, segment_encoder_type="transformer", **config_dict)
model = HierXLMRobertaForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# adapter_config = AdapterConfig.load("pfeiffer", non_linearity=None, reduction_factor=None)
# model.add_adapter(task_name, config=adapter_config)
# model.train_adapter([task_name])
# model.set_active_adapters(task_name)

def preprocess_function(batch):
    batch['segments'] = []
    tokenized = tokenizer(batch["text"], padding=padding, truncation=True,
                          max_length=max_segments * max_seg_len, add_special_tokens=False)
    for ids in tokenized['input_ids']:
        # convert ids to tokens and then back to strings
        id_blocks = [ids[i:i + max_seg_len] for i in range(0, len(ids), max_seg_len) if ids[i] != config.pad_token_id]
        token_blocks = [tokenizer.convert_ids_to_tokens(ids) for ids in id_blocks]
        string_blocks = [tokenizer.convert_tokens_to_string(tokens) for tokens in token_blocks]
        batch['segments'].append(string_blocks)

    # Tokenize the texts
    def append_zero_segments(case_encodings, pad_id=0):
        """appends a list of zero segments to the encodings to make up for missing segments"""
        return case_encodings + [[pad_id] * max_seg_len] * (max_segments - len(case_encodings))

    tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    for case in batch['segments']:
        case_encodings = tokenizer(case[:max_segments], padding=padding, truncation=True,
                                   max_length=max_seg_len, return_token_type_ids=True)
        tokenized['input_ids'].append(append_zero_segments(case_encodings['input_ids'], pad_id=config.pad_token_id))
        tokenized['attention_mask'].append(append_zero_segments(case_encodings['attention_mask'], pad_id=0))
        tokenized['token_type_ids'].append(append_zero_segments(case_encodings['token_type_ids'], pad_id=0))
    del batch['segments']

    return tokenized


batch = {"text": []}
for i in range(4):
    batch['text'].append(' a ' * max_seg_len * random.randint(1, max_segments))
batch = preprocess_function(batch)
print("tokenization successful")

outputs = model(input_ids=torch.tensor(batch['input_ids'], device=device),
                attention_mask=torch.tensor(batch['attention_mask'], device=device),
                token_type_ids=torch.tensor(batch['token_type_ids'], device=device))
print(outputs)

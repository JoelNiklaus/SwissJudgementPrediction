import faulthandler

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from HierarchicalBert import HierarchicalBert

faulthandler.enable()

max_segments = 2
max_doc_length = 1024
max_length = max_doc_length - max_segments * 2
device = 'cpu'  # 'cpu'  # 'cuda:0'

model_name = "xlm-roberta-base"
model_type = 'xlm-roberta'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if model_type == 'bert':
    encoder = model.bert

if model_type in ['camembert', 'xlm-roberta']:
    encoder = model.roberta

hier_bert = HierarchicalBert(encoder,
                             max_segments=max_segments,
                             max_segment_length=512,
                             cls_token_id=tokenizer.cls_token_id,
                             sep_token_id=tokenizer.sep_token_id,
                             device=device,
                             seg_encoder_type='lstm')

if model_type == 'bert':
    model.bert = hier_bert

if model_type in ['camembert', 'xlm-roberta']:
    dropout = nn.Dropout(model.config.hidden_dropout_prob)
    out_proj = nn.Linear(model.config.hidden_size, model.config.num_labels)
    model.classifier = nn.Sequential(dropout, out_proj)
    model.roberta = hier_bert

batch = tokenizer(['a ' * 1024] * 4, truncation=True, padding='max_length',
                  max_length=max_length, add_special_tokens=False, return_tensors='pt', return_token_type_ids=True)

# This is only necessary if return_token_type_ids is False
# if not hasattr(tokenized, 'token_type_ids'):  # RoBERTa based models do not use token_type_ids
#    tokenized['token_type_ids'] = [tokenizer.create_token_type_ids_from_sequences(input_id) for input_id in tokenized['input_ids']]
# This should be done if return_tensors is enabled
# tokenized['token_type_ids'] = torch.zeros(tokenized['input_ids'].size(), dtype=torch.int, device=training_args.device)


outputs = model(input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'])
print(outputs)

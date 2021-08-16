import faulthandler

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# IMPORTANT: This can lead to memory and performance issues with too high max_length because of the quadratic attention!
import LongBert

faulthandler.enable()

max_length = 1024
device = 'cpu'  # 'cpu'  # 'cuda:0'

# Load model and tokenizer
model_name = "xlm-roberta-base"
# model_name = 'deepset/gbert-base'
model_type = 'xlm-roberta'
# model_type = 'bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

if model_type == 'bert':
    encoder = model.bert

if model_type in ['camembert', 'xlm-roberta']:
    encoder = model.roberta

# Extend position embeddings
long_bert = LongBert.resize_position_embeddings(encoder, max_length=max_length, device=device)

if model_type == 'bert':
    model.bert = long_bert

if model_type in ['camembert', 'xlm-roberta']:
    model.roberta = long_bert

# Test
tokens = tokenizer(['a ' * max_length] * 2, truncation=True, padding='max_length', max_length=max_length,
                   return_tensors='pt', return_token_type_ids=True)
outputs = model(input_ids=tokens.data['input_ids'])
print(outputs)

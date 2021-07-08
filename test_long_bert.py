import faulthandler

from transformers import AutoTokenizer, BertForSequenceClassification

# IMPORTANT: This can lead to memory and performance issues with too high max_length because of the quadratic attention!
import LongBert

faulthandler.enable()

# Load model and tokenizer
model_name = 'deepset/gbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Extend position embeddings
model = LongBert.resize_position_embeddings(model, max_length=1024)

# Test
tokens = tokenizer(['a ' * 1024] * 2, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
outputs = model(input_ids=tokens.data['input_ids'])
print(outputs)

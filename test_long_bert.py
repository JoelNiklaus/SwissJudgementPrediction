from transformers import AutoTokenizer, BertForSequenceClassification
import torch

# IMPORTANT: This can lead to memory and performance issues with too high max_length because of the quadratic attention!
def resize_position_embeddings(model, max_length=1024):
    old_max_length = model.config.max_position_embeddings
    assert max_length % old_max_length == 0

    # Create new embedding layer
    new_embeddings = torch.nn.Embedding(max_length, model.bert.config.hidden_size).to(model.device)

    # Replicate embeddings N times
    for N in range(int(max_length / old_max_length)):
        new_embeddings.weight.data[N * old_max_length:(N + 1) * old_max_length, :] = \
            model.bert.embeddings.position_embeddings.weight.data[:old_max_length, :]
    model.bert.embeddings.position_embeddings = new_embeddings
    model.bert.embeddings.position_ids = torch.arange(max_length).expand((1, -1))

    # Fix config values
    model.config.max_position_embeddings = max_length
    model.max_position_embeddings = max_length

    return model


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-base')
model = BertForSequenceClassification.from_pretrained('deepset/gbert-base')

# Extend position embeddings
model = resize_position_embeddings(model, max_length=1024)

# Test
tokens = tokenizer(['a ' * 1024] * 2, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
outputs = model(input_ids=tokens.data['input_ids'])
print(outputs)

import copy
import torch
from transformers import AutoModelForSequenceClassification


def convert2longformer(encoder, max_seq_length: int, attention_window=128):
    supported_models = ['bert', 'camembert', 'xlm-roberta']
    assert encoder.config.model_type in supported_models  # other models are not supported so far

    # extend position embedding
    config = encoder.config
    embeddings = encoder.embeddings
    current_max_pos, embed_size = embeddings.position_embeddings.weight.shape
    max_seq_length += 2
    config.max_position_embeddings = max_seq_length
    assert max_seq_length > current_max_pos

    new_pos_embed = embeddings.position_embeddings.weight.new_empty(max_seq_length, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    new_pos_embed[:k] = embeddings.position_embeddings.weight[:k]
    step = current_max_pos - 2
    while k < max_seq_length - 1:
        if k + step >= max_seq_length:
            new_pos_embed[k:] = embeddings.position_embeddings.weight[2:(max_seq_length + 2 - k)]
        else:
            new_pos_embed[k:(k + step)] = embeddings.position_embeddings.weight[2:]
        k += step
    embeddings.position_embeddings.weight.data = new_pos_embed
    embeddings.position_ids.data = torch.tensor([i for i in range(max_seq_length)]).reshape(1, max_seq_length)

    # add global attention
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i in range(len(encoder.encoder.layer)):
        self_attention = encoder.encoder.layer[i].attention
        self_attention.self.query_global = copy.deepcopy(self_attention.self.query)
        self_attention.self.key_global = copy.deepcopy(self_attention.self.key)
        self_attention.self.value_global = copy.deepcopy(self_attention.self.value)

    # Extra config parameters
    encoder.config.attention_mode = "longformer"
    encoder.config.model_type = "longformer"

    return encoder


if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained("deepset/gbert-base")
    model = convert2longformer(model, max_seq_length=2048)

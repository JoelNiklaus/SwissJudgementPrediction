import torch


def resize_position_embeddings(encoder, max_length, device):
    supported_models = ['distilbert', 'bert', 'camembert', 'xlm-roberta']
    assert encoder.config.model_type in supported_models  # other models are not supported so far

    old_max_length = encoder.config.max_position_embeddings
    old_max_length = 512  # because RoBERTa has 514 max_seq_length and not 512
    assert max_length % old_max_length == 0

    # Create new embedding layer
    embedding_length = max_length
    if encoder.config.model_type in ['camembert', 'xlm-roberta']:
        embedding_length += 2  # because RoBERTa is FUCKING STUPID (position ids start at 2 instead of 0)
    new_embeddings = torch.nn.Embedding(embedding_length, encoder.config.hidden_size).to(device)

    # Replicate embeddings N times
    for N in range(int(embedding_length / old_max_length)):
        new_embeddings.weight.data[N * old_max_length:(N + 1) * old_max_length, :] = \
            encoder.embeddings.position_embeddings.weight.data[:old_max_length, :]
    if encoder.config.model_type in ['camembert', 'xlm-roberta']:
        # Fill in the last two weights because the embedding_length is higher for RoBERTa
        new_embeddings.weight.data[-2:] = encoder.embeddings.position_embeddings.weight.data[-2:, :]
    encoder.embeddings.position_embeddings = new_embeddings
    encoder.embeddings.position_ids = torch.arange(embedding_length).expand((1, -1)).to(device)
    encoder.embeddings.token_type_ids = torch.zeros(embedding_length, dtype=torch.int).expand((1, -1)).to(device)
    # Fix config values
    encoder.config.max_position_embeddings = embedding_length
    encoder.max_position_embeddings = embedding_length

    return encoder

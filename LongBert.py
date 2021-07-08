import torch


def resize_position_embeddings(encoder, max_length):
    supported_models = ['bert', 'camembert', 'xlm-roberta']
    assert encoder.config.model_type in supported_models  # other models are not supported so far

    old_max_length = encoder.config.max_position_embeddings
    old_max_length = 512  # because RoBERTa has 514 max_seq_length and not 512
    assert max_length % old_max_length == 0

    # Create new embedding layer
    new_embeddings = torch.nn.Embedding(max_length, encoder.config.hidden_size).to(encoder.device)

    # Replicate embeddings N times
    for N in range(int(max_length / old_max_length)):
        new_embeddings.weight.data[N * old_max_length:(N + 1) * old_max_length, :] = \
            encoder.embeddings.position_embeddings.weight.data[:old_max_length, :]
    encoder.embeddings.position_embeddings = new_embeddings
    encoder.embeddings.position_ids = torch.arange(max_length).expand((1, -1))

    # Fix config values
    encoder.config.max_position_embeddings = max_length
    encoder.max_position_embeddings = max_length

    return encoder

import pandas as pd


# TODO experiment with randomly initialized transformer
# TODO do we need to experiment with a BiLSTM model?

# TODO try using data augmentation: https://arxiv.org/abs/2105.03075

model_names = {  # distilbert-base-multilingual-cased, google/rembert (not yet in adapter-transformers), google/mt5-base (not yet in adapter-transformers),
    'de': ['xlm-roberta-base', 'deepset/gbert-base'],
    'fr': ['xlm-roberta-base', 'camembert/camembert-base-ccnet'],
    'it': ['xlm-roberta-base', 'Musixmatch/umberto-commoncrawl-cased-v1'],  # dbmdz/bert-base-italian-cased
    'all': ['xlm-roberta-base', 'bert-base-multilingual-cased'],
    # 'en': ['google/bigbird-roberta-base', 'bert-base-cased', 'roberta-base'],
}
types = ['standard', 'hierarchical', 'long']  # longformer
languages = ['de', 'fr', 'it', 'all']
train_languages = ['de', 'all']

train_experiments = []
for lang in languages:
    for type in types:
        for model_name in model_names[lang]:
            experiment = {"model_name": model_name, "type": type, "lang": lang, "status": ""}
            train_experiments.append(experiment)
df = pd.DataFrame(train_experiments)
df.to_csv("train_experiments.csv", index=False)

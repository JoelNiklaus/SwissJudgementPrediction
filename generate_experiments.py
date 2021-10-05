import pandas as pd

# TODO implement baselines with BiLSTM, bow, tf-idf
# TODO experiment with randomly initialized transformer

# TODO try using data augmentation: https://arxiv.org/abs/2105.03075

# TODO work with hierarchical model
# TODO use new splitting for hierarchical bert
#       319: https://github.com/coastalcph/lex-glue/blob/main/experiments/ecthr.py
#           https://github.com/coastalcph/lex-glue/blob/main/models/hierbert.py
# Split with this code:
# sentences = []
# for sent in sent_tokenize(example['text'], language=ISO2LANGUAGE[example['language']]):
# 	if (len(sent) <= 50 or re.match('[0-9]', sent)) and len(sentences):
# 		sentences[-1] += ' ' + sent
# 	else:
# 		sentences.append(sent)
# ==> test on german and see if this improves results
# ==> investigate average paragraph length to see if it makes sense to use these
# TODO train on all languages, evaluate on each language
# TODO train on other languages, evaluate on third language ==> zero shot
# TODO ECTHR violation ==> approval, non-violation ==> dismissal, binarize SCOTUS as well, compare with just further pretraining on this data
# TODO translate all cases to one language and evaluate on that language with native model
# TODO use paraphrases/backtranslation as data augmentation


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

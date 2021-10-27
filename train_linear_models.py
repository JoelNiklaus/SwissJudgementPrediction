from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

import os, tqdm, json
from data import DATA_DIR

LANGUAGE = 'de'
scores = {LANGUAGE: {'micro-f1': [], 'macro-f1': []} for LANGUAGE in ['de', 'fr', 'it']}
for LANGUAGE in ['de', 'fr', 'it']:
    for seed in range(1, 6):
        if LANGUAGE == 'all':
            stop_words = set(stopwords.words('german') + stopwords.words('french') + stopwords.words('italian'))
        elif LANGUAGE == 'de':
            stop_words = stopwords.words('german')
        elif LANGUAGE == 'fr':
            stop_words = stopwords.words('french')
        elif LANGUAGE == 'it':
            stop_words = stopwords.words('italian')

        texts = {'train': [], 'val': [], 'test': []}
        targets = {'train': [], 'val': [], 'test': []}
        for subset in ['train', 'val', 'test']:
            with open(os.path.join(DATA_DIR, 'datasets', f'fscs_v1.0', f'{subset}.jsonl')) as file:
                for row in tqdm.tqdm(file.readlines()):
                    example = json.loads(row)
                    if example['language'] == LANGUAGE:
                        texts[subset].append(example['text'])
                        targets[subset].append(1 if example['label'] == 'approval' else 0)

        # texts['train'].extend(texts['val'])
        # targets['train'].extend(targets['val'])
        # weights = compute_class_weight('balanced',classes=np.unique(targets['train']),y=targets['train'])
        # weights = {0: weights}
        from sklearn.pipeline import Pipeline
        text_clf = Pipeline([('vect', CountVectorizer(stop_words=stop_words, ngram_range=(1, 3))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(early_stopping=True, learning_rate='adaptive',
                                                   eta0=1e-4, validation_fraction=0.1, max_iter=10000,
                                                   class_weight='balanced', random_state=seed)),
                             ])

        parameters = {
            'vect__max_features': [5000, 10000, 20000, 35000],
            'tfidf__use_idf': (True, False),
            # 'clf__alpha': (1e-3, 1e-4),
            'clf__loss': ('hinge', 'log')
        }

        gs_clf = GridSearchCV(text_clf, parameters, cv=None, n_jobs=-1, verbose=4)

        gs_clf = gs_clf.fit(texts['train'], targets['train'])

        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

        print('VALIDATION RESULTS:')
        preds = gs_clf.predict(texts['val'])
        print(f'Micro-F1: {metrics.f1_score(targets["val"], preds, average="micro")*100:.1f}')
        print(f'Macro-F1: {metrics.f1_score(targets["val"], preds, average="macro")*100:.1f}')
        print('TEST RESULTS:')
        preds = gs_clf.predict(texts['test'])
        print(f'Micro-F1: {metrics.f1_score(targets["test"], preds, average="micro")*100:.1f}')
        print(f'Macro-F1: {metrics.f1_score(targets["test"], preds, average="macro")*100:.1f}')

        scores[LANGUAGE]['micro-f1'].append(metrics.f1_score(targets["test"], preds, average="micro"))
        scores[LANGUAGE]['macro-f1'].append(metrics.f1_score(targets["test"], preds, average="macro"))

print('-'*100)
for LANGUAGE in ['de', 'fr', 'it']:
    print(f'{LANGUAGE}:\t Micro-F1: {np.mean(scores[LANGUAGE]["micro-f1"])*100:.1f} +/- {np.std(scores[LANGUAGE]["micro-f1"])*100:.1f}\t'
          f'Macro-F1: {np.mean(scores[LANGUAGE]["macro-f1"])*100:.1f} +/- {np.std(scores[LANGUAGE]["macro-f1"])*100:.1f}')
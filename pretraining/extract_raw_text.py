import json
import random
import numpy as np
from tqdm import tqdm

from root import DATA_DIR
from utils.sentencizer import combine_small_sentences, nltk_sentencize, spacy_sentencize

filenames = ['train.jsonl', 'val.jsonl']


def extract_raw_text(language: str):
    samples, sentence_lengths = [], []
    for filename in filenames:
        print(f"Processing {filename}")
        with open(DATA_DIR / filename) as file:
            for line in tqdm(file.readlines()):
                example = json.loads(line)
                if not language or example['language'] == language:
                    sents = nltk_sentencize(example['text'], example['language'])
                    # sents = spacy_sentencize(example['text'], example['language']) # quite a bit slower
                    sentences = combine_small_sentences(sents, 50)
                    samples.extend(sentences)
                    sentence_lengths.append(len(sentences))

    print(f'{np.mean(sentence_lengths):.1f} +/- {np.std(sentence_lengths):.1f}')

    random.shuffle(samples)
    file_name = f'fscs_dump_{language}.txt' if language else 'fscs_dump.txt'
    with open(DATA_DIR / file_name, 'w') as out_file:
        for sample in samples:
            out_file.write(sample.replace('\n', ' ') + '\n')


for language in ['de', 'fr', 'it']:
    extract_raw_text(language)

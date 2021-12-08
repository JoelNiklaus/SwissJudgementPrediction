import json
import random
import numpy as np
from tqdm import tqdm

from utils.sentencizer import combine_small_sentences, nltk_sentencize, spacy_sentencize

filenames = ['train.jsonl', 'val.jsonl']
samples = []
sentence_lengths = []

for filename in filenames:
    print(f"Processing {filename}")
    with open('data/' + filename) as file:
        for line in tqdm(file.readlines()):
            example = json.loads(line)
            sents = nltk_sentencize(example['text'], example['language'])
            # sents = spacy_sentencize(example['text'], example['language']) # quite a bit slower
            sentences = combine_small_sentences(sents, 50)
            samples.extend(sentences)
            sentence_lengths.append(len(sentences))

print(f'{np.mean(sentence_lengths):.1f} +/- {np.std(sentence_lengths):.1f}')

random.shuffle(samples)
with open('data/fscs_dump.txt', 'w') as out_file:
    for sample in samples:
        out_file.write(sample.replace('\n', ' ') + '\n')

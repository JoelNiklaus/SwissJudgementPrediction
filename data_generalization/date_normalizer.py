"""
Run this script from the root folder like this: python -m data_generalization.date_normalizer
"""

from root import DATA_DIR, DATE_NORMALIZATION_DIR
from pathlib import Path
from shutil import copyfile
import pandas as pd
from tqdm import tqdm
import re


# dict containing all the months in the languages german, french and italian
months = {
    'de': ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember'],
    'fr': ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'],
    'it': ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
}

# dict containing the word "year" in the languages german, french and italian
year_word = {
    'de': 'Jahr',
    'fr': 'année',
    'it': 'anno'
}

# dict containing the word "month" in the languages german, french and italian
month_word = {
    'de': 'Monat',
    'fr': 'mois',
    'it': 'mese'
}

# dict containing the word "date" in the languages german, french and italian
date_word = {
    'de': 'Datum',
    'fr': 'date',
    'it': 'data'
}


# Checking the text string for dates with regex and replace them with the normalized dates
# Format only for Italian data at the moment
def replace_date(text, lang):
    for month in months[lang]:
        text = re.sub(r'\d{1,2}°? ' + month + r' ?.{0,4}(?:\d{4})?', date_word[lang], text)     # type 1: e.g. 29 maggio (del) 1935 -> data
        text = re.sub(month + r' ?.{0,4}\d{4}', date_word[lang], text)                          # type 2: e.g. maggio (del) 1935 -> data
        text = re.sub(month, month_word[lang], text)                                            # type 3: e.g. maggio -> mese
    text = re.sub(r'\d{1,2}.\d{1,2}.\d{4}', date_word[lang], text)                              # type 4: e.g. 29.02.1935 -> data
    text = re.sub(r'(?<!n\. )\d{4}', year_word[lang], text)                                     # type 5: e.g. 1935 -> anno but not n. 5321
    text = re.sub(date_word[lang] + ' ' + date_word[lang], date_word[lang], text)       # replace 'date date' with 'date'
    text = re.sub(year_word[lang] + ' ' + year_word[lang], year_word[lang], text)       # replace 'anno anno' with 'anno'
    text = re.sub('mese di data', 'data', text)                                         # replace 'mese di data' with 'data'
    text = re.sub('mese di mese', 'mese', text)                                         # replace 'mese di data' with 'mese'
    return text


def normalize_texts(lang, texts):
    normalized_texts_list = []
    for text in tqdm(texts):
        normalized = replace_date(text, lang)
        normalized_texts_list.append(normalized)
    return normalized_texts_list


def normalize(source_langs):
    target_path = DATE_NORMALIZATION_DIR
    target_path.mkdir(parents=True, exist_ok=True)

    for lang in source_langs:
        copyfile(f'{DATA_DIR}/{lang}/labels.json', f'{target_path}/labels.json')  # copy labels.json file
        file_path = f'{DATA_DIR}/{lang}/train.csv'
        target_file_path = target_path / f'train_{lang}.csv'
        if Path(target_file_path).exists():
            print(f"Already processed {target_file_path}. Skipping...")
            continue
        print(f"Reading data from {file_path}")
        df = pd.read_csv(file_path, nrows=2 if debug else None)

        # normalize dates from the 'lang' dataset
        print(f"Normalizing {lang} dataset")

        df.text = normalize_texts(lang, df.text.tolist())

        print(f"Saving data to {target_file_path}")
        df.to_csv(target_file_path)


if __name__ == '__main__':
    debug = False
    source_langs = ['it']   # ['de', 'fr', 'it']
    normalize(source_langs)


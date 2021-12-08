"""
Run this script from the root folder like this: python -m data_augmentation.translate

(a) use google translate with googletrans python library, using sleep calls in-between whenever you exceed the free quota
(b) use openNMT (https://opennmt.net)
(c) use huggingface pipelines:
    - https://huggingface.co/Helsinki-NLP/opus-mt-de-it
    - https://huggingface.co/Helsinki-NLP/opus-mt-de-fr
    - https://huggingface.co/Helsinki-NLP/opus-mt-it-de
    - https://huggingface.co/Helsinki-NLP/opus-mt-it-fr
    - https://huggingface.co/Helsinki-NLP/opus-mt-fr-de
    - https://huggingface.co/Helsinki-NLP/opus-tatoeba-fr-it
(d) - use easynmnt
"""
from root import ROOT_DIR
from pathlib import Path
from shutil import copyfile
import pandas as pd
from tqdm import tqdm
from easynmt import EasyNMT

data_dir = ROOT_DIR / 'data'
augmented_path = data_dir / 'augmented'
translated_path = augmented_path / 'translated'
back_translated_path = augmented_path / 'back_translated'
languages = ['de', 'fr', 'it', 'es', 'en']
# the translations seem ok. Sometimes they seem to have some mistakes.
model = EasyNMT('m2m_100_418M')  # RTX 3090 is not big enough for m2m_100_1.2B. opus-mt does not support fr-it and it-he


# IMPORTANT: Use separate conda env (data_aug), because it installs transformers library (which messes with the adapter-transformers library!)

# TODO save translated texts in batches to avoid timeout problems (35K documents take more than 24h)


def translate_texts(source_lang, target_lang, texts):
    translated_texts_list = []
    for text in tqdm(texts):
        translated = model.translate(text, source_lang=source_lang, target_lang=target_lang, batch_size=8)
        translated_texts_list.append(translated)
    return translated_texts_list


def translate(source_langs, target_lang, debug=False) -> None:
    target_path = translated_path / target_lang
    target_path.mkdir(parents=True, exist_ok=True)
    copyfile(f'{data_dir}/{target_lang}/labels.json', f'{target_path}/labels.json')  # copy labels.json file

    for lang in source_langs:
        file_path = f'{data_dir}/{lang}/train.csv'
        target_file_path = target_path / f'train_{lang}.csv'
        if Path(target_file_path).exists():
            print(f"Already processed {target_file_path}. Skipping...")
            continue
        print(f"Reading data from {file_path}")
        df = pd.read_csv(file_path, nrows=2 if debug else None)

        # translate into target language
        print(f"Translating {lang} into {target_lang}")

        df.text = translate_texts(lang, target_lang, df.text.tolist())
        df['source_language'] = lang

        print(f"Saving data to {target_file_path}")
        df.to_csv(target_file_path)


def back_translate(source_lang, target_langs, debug=False) -> None:
    # https://huggingface.co/blog/how-to-generate We could also play with different temperatures, etc.
    target_path = back_translated_path / source_lang
    target_path.mkdir(parents=True, exist_ok=True)
    copyfile(f'{data_dir}/{source_lang}/labels.json', f'{target_path}/labels.json')  # copy labels.json file

    for lang in target_langs:
        file_path = f'{data_dir}/{source_lang}/train.csv'
        target_file_path = target_path / f'train_{lang}.csv'
        if Path(target_file_path).exists():
            print(f"Already processed {target_file_path}. Skipping...")
            continue
        print(f"Reading data from {file_path}")
        df = pd.read_csv(file_path, nrows=2 if debug else None)

        # translate into target language
        print(f"Translating {source_lang} into {lang}")
        translated_texts = translate_texts(source_lang, lang, df.text.tolist())

        # translate back into source language
        print(f"Translating {lang} back into {source_lang}")
        df.text = translate_texts(lang, source_lang, translated_texts)
        df['destination_language'] = lang

        print(f"Saving data to {target_file_path}")
        df.to_csv(target_file_path)


if __name__ == '__main__':
    debug = False

    source_langs = ['de', 'fr']
    target_lang = 'it'
    translate(source_langs, target_lang, debug=debug)

    # TODO find reason for a number of languages
    source_lang = 'it'
    target_langs_basic = ['de', 'fr', 'en', 'es', 'pt']
    opus_mt_langs = ['de', 'fr', 'es', 'en', 'he', 'ar', 'bg', 'ca', 'eo', 'is', 'lt', 'ms', 'uk',
                     'vi', ]  # supported by opus-mt
    m2m_100_langs = ['af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy', 'da',
                     'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he',
                     'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn',
                     'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl',
                     'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq',
                     'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh',
                     'yi', 'yo', 'zh', 'zu']  # supported by m2m_100
    back_translate(source_lang, target_langs_basic, debug=debug)

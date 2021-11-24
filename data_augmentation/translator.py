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
"""
from shutil import copyfile
import pandas as pd
from transformers import pipeline

from root import ROOT_DIR
from utils.sentencizer import get_sentencizer

data_dir = ROOT_DIR / 'data'
augmented_path = data_dir / 'augmented'
translated_path = augmented_path / 'translated'
back_translated_path = augmented_path / 'back_translated'
languages = ['de', 'fr', 'it']

# TODO check if the translations make sense
# TODO translate by paragraph, otherwise there seems to be a problem (too many characters)
sentencizers = {lang: get_sentencizer(lang) for lang in languages}


def sentencize(lang, texts):
    sents_list = []
    nlp = sentencizers[lang]
    for doc in nlp.pipe(texts, batch_size=len(texts)):
        sents_list.append([sent.text for sent in doc.sents])
    return sents_list


def setup_translator(source_lang, dest_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}"
    if (source_lang == 'fr' and dest_lang == 'it') or (source_lang == 'it' and dest_lang == 'he'):
        model_name = f"Helsinki-NLP/opus-tatoeba-{source_lang}-{dest_lang}"
    return pipeline('translation', model=model_name)


def run_translator(translator, texts):
    # Make sure, not to pass in inputs longer than 512 tokens / approx. 400 words
    translated = [text['translation_text'] for text in translator(texts)]
    return translated


def translate(source_langs, dest_lang, debug=False) -> None:
    dest_translated_path = translated_path / dest_lang
    dest_translated_path.mkdir(parents=True, exist_ok=True)

    for lang in source_langs:
        df = pd.read_csv(f'{data_dir}/{lang}/train.csv', nrows=2 if debug else None)
        texts = df.text.tolist()
        sents_list = sentencize(lang, texts)

        # translate into destination language
        print(f"Translating {lang} into {dest_lang}")
        translated_sents_list = []
        translator = setup_translator(lang, dest_lang)
        for sents in sents_list:
            translated_sents_list.append(run_translator(translator, sents))
        print(translated_sents_list)

        df.text = [" ".join(sents) for sents in translated_sents_list]
        df['source_language'] = lang

        df.to_csv(dest_translated_path / f'train_{lang}.csv')
    copyfile(f'{data_dir}/{dest_lang}/labels.json', f'{dest_translated_path}/labels.json')  # copy labels.json file


def back_translate(source_lang, dest_langs, debug=False) -> None:
    # https://huggingface.co/blog/how-to-generate We could also play with different temperatures, etc.
    dest_back_translated_path = back_translated_path / source_lang
    dest_back_translated_path.mkdir(parents=True, exist_ok=True)

    for lang in dest_langs:
        df = pd.read_csv(f'{data_dir}/{source_lang}/train.csv', nrows=2 if debug else None)
        texts = df.text.tolist()
        sents_list = sentencize(lang, texts)

        # translate into destination language
        print(f"Translating {source_lang} into {lang}")
        translated_sents_list = []
        translator = setup_translator(source_lang, lang)
        for sents in sents_list:
            translated_sents_list.append(run_translator(translator, sents))

        # translate back into source language
        print(f"Translating {lang} back into {source_lang}")
        back_translated_sents_list = []
        back_translator = setup_translator(lang, source_lang)
        for sents in translated_sents_list:
            back_translated_sents_list.append(run_translator(back_translator, sents))  #
        print(back_translated_sents_list)

        df.text = [" ".join(sents) for sents in back_translated_sents_list]
        df['destination_language'] = lang

        df.to_csv(dest_back_translated_path / f'train_{lang}.csv')

    copyfile(f'{data_dir}/{source_lang}/labels.json',
             f'{dest_back_translated_path}/labels.json')  # copy labels.json file


if __name__ == '__main__':
    debug = False

    source_langs = ['de', 'fr']
    dest_lang = 'it'
    translate(source_langs, dest_lang, debug=debug)

    source_lang = 'it'
    dest_langs_short = ['de', 'fr', 'es', 'en', ]
    dest_langs_long = ['de', 'fr', 'es', 'en', 'he', 'ar', 'bg', 'ca', 'eo', 'is', 'lt', 'ms', 'uk', 'vi', ]
    back_translate(source_lang, dest_langs_short, debug=debug)

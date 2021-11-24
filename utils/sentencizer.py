def get_sentencizer(lang):
    if lang == 'de':
        from spacy.lang.de import German
        nlp = German()
    elif lang == 'fr':
        from spacy.lang.fr import French
        nlp = French()
    elif lang == 'it':
        from spacy.lang.it import Italian
        nlp = Italian()
    else:
        raise ValueError(f"The language {lang} is not supported.")
    nlp.add_pipe("sentencizer")
    return nlp

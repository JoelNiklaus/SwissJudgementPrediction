import re

from spacy import Language
from typing import List
from nltk import sent_tokenize


def nltk_sentencize(text: str, language: str) -> List[str]:
    langs = {'de': 'german', 'fr': 'french', 'it': 'italian'}
    if language not in langs:
        raise ValueError(f"The language {language} is not supported.")
    return sent_tokenize(text, language=langs[language])


def get_spacy_sents(text: str, nlp: Language) -> List[str]:
    return [sent.text for sent in nlp(text).sents]


def spacy_sentencize(text: str, language: str) -> List[str]:
    sentencizer = get_sentencizer(language)
    return get_spacy_sents(text, sentencizer)


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
    elif lang == 'en':
        from spacy.lang.en import English
        nlp = English()
    elif lang == 'es':
        from spacy.lang.es import Spanish
        nlp = Spanish()
    elif lang == 'pt':
        from spacy.lang.pt import Portuguese
        nlp = Portuguese()
    else:
        raise ValueError(f"The language {lang} is not supported.")
    nlp.add_pipe("sentencizer")
    return nlp


def combine_small_sentences(sents, min_sentence_length):
    sentences = []
    for sent in sents:
        if (len(sent) <= min_sentence_length or re.match('[0-9]', sent)) and len(sentences):
            sentences[-1] += ' ' + sent
        else:
            sentences.append(sent)
    return sentences


def test_performance(type: str):
    example_document = """
    Sachverhalt: A. A.a. A._ schloss mit der Berner Lebensversicherungs-Gesellschaft (seit 13. September 2009: Allianz Suisse Schweizerische Lebensversicherungs-Gesellschaft AG; nachfolgend: Allianz) im Jahre 1995 einen Vertrag \u00fcber eine Einzel-Lebensversicherung im Rahmen der gebundenen Vorsorge der S\u00e4ule 3a ab (Police vom 18. August 1995). Mit Wirkung auf 1. Mai 2000 wurde dieser Vertrag ersetzt. Nach der dazugeh\u00f6renden Police Nr. ... hat A._ bei Erwerbsunf\u00e4higkeit ab dem 721. Tag Anspruch auf eine Rente von Fr. 24'000.- pro Jahr und ab dem 91. Tag Anspruch auf Pr\u00e4mienbefreiung. A.b. Vom 1. Juli bis 31. Dezember 1998 bezog A._ aufgrund eines chronischen Thorako- und Lumbovertebralsyndroms eine (befristete) halbe Rente der Invalidenversicherung (Verf\u00fcgung der IV-Stelle Basel-Landschaft [nachfolgend: IV-Stelle] vom 9. Dezember 1999). Auf seine Neuanmeldung vom Dezember 2001 hin sprach ihm die IV-Stelle f\u00fcr die Zeit vom 1. M\u00e4rz bis 31. Mai 2001 eine halbe (Invalidit\u00e4tsgrad: 50 %) und ab 1. Juni 2001 eine ganze Invalidenrente (Invalidit\u00e4tsgrad: 73 %) zu (Verf\u00fcgungen vom 24. September 2003 und 20. Juli 2004). Mit Mitteilung vom 16. Januar 2007 best\u00e4tigte die IV-Stelle den Anspruch revisionsweise. Die Allianz anerkannte ihre Leistungspflicht und erbrachte Leistungen entsprechend dem jeweiligen Erwerbsunf\u00e4higkeitsgrad (100 % vom 21. M\u00e4rz bis 12. August 2001, 50 % vom 13. August bis 2. September 2001 und 100 % ab 3. September 2001). Im Rahmen eines 2012 eingeleiteten Revisionsverfahrens beauftragte die IV-Stelle Dr. med. B._, FMH Rheumatologie sowie Physikalische Medizin und Rehabilitation, den Versicherten zu begutachten (Gutachten vom 23. Juli 2012; Erg\u00e4nzungsgutachten vom 19. Dezember 2012). Nach Einholung einer Stellungnahme beim Regionalen \u00e4rztlichen Dienst (RAD) (erstattet am 10. Januar 2013) verneinte die IV-Stelle das Vorliegen eines Revisionsgrundes. Sie teilte A._ mit, dass er unver\u00e4ndert Anspruch auf eine ganze Invalidenrente habe (Mitteilung vom 31. Januar 2013). A.c. Am 10. April 2013 stellte die Allianz A._ die Schlussabrechnung zu, unter Hinweis darauf, dass sie bis zum 31. Januar 2013 eine Erwerbsunf\u00e4higkeit von 100 % anerkenne und ihre Leistungen mit Wirkung auf 1. Februar 2013 einstelle. In einem weiteren Schreiben vom 17. April 2013 informierte sie ihn dar\u00fcber, dass sie gest\u00fctzt auf das Gutachten des Dr. med. B._ vom 23. Juli 2012 von einer Arbeitsf\u00e4higkeit von 80 % und damit von einer Verbesserung des Gesundheitszustandes ausgehe. B. Klageweise liess A._ beantragen, die Allianz sei zu verpflichten, ihm \u00fcber den 1. Februar 2013 hinaus die in der Einzel-Lebensversicherungspolice vorgesehene j\u00e4hrliche Rente von Fr. 24'000.- einschliesslich Zins von 5 % ab F\u00e4lligkeit zu erbringen und die volle Pr\u00e4mienbefreiung zu gew\u00e4hren. Das Kantonsgericht Basel-Landschaft hiess die Klage gut und verpflichtete die Allianz, A._ \u00fcber den 1. Februar 2013 hinaus weiterhin aus der Einzel-Lebensversicherungspolice Nr. ... eine j\u00e4hrliche Rente von Fr. 24'000.- zu erbringen und ihn von der Zahlung der Pr\u00e4mien von Fr. 6'917.- pro Jahr zu befreien. Sie habe die nachzuzahlenden Renten ab 31. M\u00e4rz 2013 und allf\u00e4llige zur\u00fcckzuerstattende Pr\u00e4mien ab 19. Juni 2013 zu 5 % zu verzinsen (Entscheid vom 27. Februar 2014). C. Die Allianz f\u00fchrt Beschwerde in \u00f6ffentlich-rechtlichen Angelegenheiten mit dem Rechtsbegehren, der kantonale Entscheid sei aufzuheben. Die Klage sei abzuweisen. Eventualiter sei die Sache an die Vorinstanz zur\u00fcckzuweisen, damit sie pr\u00fcfe, ob und in welchem Ausmass die vertraglichen Leistungsvoraussetzungen ab 1. Februar 2013 erf\u00fcllt sind und ob A._ verneinendenfalls eine angemessene Anpassungsfrist bis zur Leistungseinstellung zuzugestehen ist. A._ l\u00e4sst auf Abweisung der Beschwerde schliessen. Das Bundesamt f\u00fcr Sozialversicherungen verzichtet auf eine Vernehmlassung.
    """
    if type == 'nltk':
        return nltk_sentencize(example_document, 'de')
    elif type == 'spacy':
        return spacy_sentencize(example_document, 'de')
    else:
        raise ValueError(f"The type {type} is not supported.")


def test_quality(type: str):
    example_document = """
    Sachverhalt: A. A.a. A._ schloss mit der Berner Lebensversicherungs-Gesellschaft (seit 13. September 2009: Allianz Suisse Schweizerische Lebensversicherungs-Gesellschaft AG; nachfolgend: Allianz) im Jahre 1995 einen Vertrag \u00fcber eine Einzel-Lebensversicherung im Rahmen der gebundenen Vorsorge der S\u00e4ule 3a ab (Police vom 18. August 1995). Mit Wirkung auf 1. Mai 2000 wurde dieser Vertrag ersetzt. Nach der dazugeh\u00f6renden Police Nr. ... hat A._ bei Erwerbsunf\u00e4higkeit ab dem 721. Tag Anspruch auf eine Rente von Fr. 24'000.- pro Jahr und ab dem 91. Tag Anspruch auf Pr\u00e4mienbefreiung. A.b. Vom 1. Juli bis 31. Dezember 1998 bezog A._ aufgrund eines chronischen Thorako- und Lumbovertebralsyndroms eine (befristete) halbe Rente der Invalidenversicherung (Verf\u00fcgung der IV-Stelle Basel-Landschaft [nachfolgend: IV-Stelle] vom 9. Dezember 1999). Auf seine Neuanmeldung vom Dezember 2001 hin sprach ihm die IV-Stelle f\u00fcr die Zeit vom 1. M\u00e4rz bis 31. Mai 2001 eine halbe (Invalidit\u00e4tsgrad: 50 %) und ab 1. Juni 2001 eine ganze Invalidenrente (Invalidit\u00e4tsgrad: 73 %) zu (Verf\u00fcgungen vom 24. September 2003 und 20. Juli 2004). Mit Mitteilung vom 16. Januar 2007 best\u00e4tigte die IV-Stelle den Anspruch revisionsweise. Die Allianz anerkannte ihre Leistungspflicht und erbrachte Leistungen entsprechend dem jeweiligen Erwerbsunf\u00e4higkeitsgrad (100 % vom 21. M\u00e4rz bis 12. August 2001, 50 % vom 13. August bis 2. September 2001 und 100 % ab 3. September 2001). Im Rahmen eines 2012 eingeleiteten Revisionsverfahrens beauftragte die IV-Stelle Dr. med. B._, FMH Rheumatologie sowie Physikalische Medizin und Rehabilitation, den Versicherten zu begutachten (Gutachten vom 23. Juli 2012; Erg\u00e4nzungsgutachten vom 19. Dezember 2012). Nach Einholung einer Stellungnahme beim Regionalen \u00e4rztlichen Dienst (RAD) (erstattet am 10. Januar 2013) verneinte die IV-Stelle das Vorliegen eines Revisionsgrundes. Sie teilte A._ mit, dass er unver\u00e4ndert Anspruch auf eine ganze Invalidenrente habe (Mitteilung vom 31. Januar 2013). A.c. Am 10. April 2013 stellte die Allianz A._ die Schlussabrechnung zu, unter Hinweis darauf, dass sie bis zum 31. Januar 2013 eine Erwerbsunf\u00e4higkeit von 100 % anerkenne und ihre Leistungen mit Wirkung auf 1. Februar 2013 einstelle. In einem weiteren Schreiben vom 17. April 2013 informierte sie ihn dar\u00fcber, dass sie gest\u00fctzt auf das Gutachten des Dr. med. B._ vom 23. Juli 2012 von einer Arbeitsf\u00e4higkeit von 80 % und damit von einer Verbesserung des Gesundheitszustandes ausgehe. B. Klageweise liess A._ beantragen, die Allianz sei zu verpflichten, ihm \u00fcber den 1. Februar 2013 hinaus die in der Einzel-Lebensversicherungspolice vorgesehene j\u00e4hrliche Rente von Fr. 24'000.- einschliesslich Zins von 5 % ab F\u00e4lligkeit zu erbringen und die volle Pr\u00e4mienbefreiung zu gew\u00e4hren. Das Kantonsgericht Basel-Landschaft hiess die Klage gut und verpflichtete die Allianz, A._ \u00fcber den 1. Februar 2013 hinaus weiterhin aus der Einzel-Lebensversicherungspolice Nr. ... eine j\u00e4hrliche Rente von Fr. 24'000.- zu erbringen und ihn von der Zahlung der Pr\u00e4mien von Fr. 6'917.- pro Jahr zu befreien. Sie habe die nachzuzahlenden Renten ab 31. M\u00e4rz 2013 und allf\u00e4llige zur\u00fcckzuerstattende Pr\u00e4mien ab 19. Juni 2013 zu 5 % zu verzinsen (Entscheid vom 27. Februar 2014). C. Die Allianz f\u00fchrt Beschwerde in \u00f6ffentlich-rechtlichen Angelegenheiten mit dem Rechtsbegehren, der kantonale Entscheid sei aufzuheben. Die Klage sei abzuweisen. Eventualiter sei die Sache an die Vorinstanz zur\u00fcckzuweisen, damit sie pr\u00fcfe, ob und in welchem Ausmass die vertraglichen Leistungsvoraussetzungen ab 1. Februar 2013 erf\u00fcllt sind und ob A._ verneinendenfalls eine angemessene Anpassungsfrist bis zur Leistungseinstellung zuzugestehen ist. A._ l\u00e4sst auf Abweisung der Beschwerde schliessen. Das Bundesamt f\u00fcr Sozialversicherungen verzichtet auf eine Vernehmlassung.
    """
    if type == 'nltk':
        return nltk_sentencize(example_document, 'de')
    elif type == 'spacy':
        return spacy_sentencize(example_document, 'de')
    else:
        raise ValueError(f"The type {type} is not supported.")


if __name__ == '__main__':
    import timeit

    sentences = test_quality('nltk')
    print("nltk: ", sentences)
    sentences = test_quality('spacy')
    print("spacy: ", sentences)

    # Qualitative Analysis: Spacy is maybe a bit better, but they are largely identical (analysis performed in German)

    number = 10
    setup = "from __main__ import test_performance"
    time = timeit.repeat("test_performance('nltk')", setup, number=number)
    print("nltk: ", time)
    print("nltk: ", min(time))
    time = timeit.repeat("test_performance('spacy')", setup, number=number)
    print("spacy: ", time)
    print("spacy: ", min(time))

    # Quantitative Analysis: NLTK is much faster (~x30: 0.025 vs 0.806)

import re
import Settings
from pymystem3 import Mystem
from typing import List, Any
from nltk.tokenize import sent_tokenize


# text processing to lemmas - sentences
def lemmatize(text_list: List[str]):
    result = list()
    for element in text_list:
        tokenized = sent_tokenize(element, language=Settings.LANGUAGE)
        for sentence in tokenized:
            normalized = normalize_sentence(sentence)
            result.append(normalized)
    return result


# removing stop words and other things from sentence
def normalize_sentence(sentence: str):
    pattern = re.compile(Settings.LEMMA_REGEX)
    mystem = Mystem()
    lemmas = mystem.lemmatize(sentence)
    result: List[str] = list()
    for word in lemmas:
        if (pattern.search(word) is not None) and (word not in Settings.STOP_WORDS):
            result.append(word)
    return result

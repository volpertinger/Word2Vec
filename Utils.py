import re
import Settings
from pymystem3 import Mystem
from typing import List, Any


def lemmatize(text_list: List[str]):
    pattern = re.compile(Settings.LEMMA_REGEX)
    result: List[Any] = list()
    mystem = Mystem()
    for element in text_list:
        lemmas = mystem.lemmatize(element)
        for lemma in lemmas:
            if pattern.search(lemma) is not None:
                result.append(lemma)
    return result


def lemmatize_(text: str):
    m = Mystem()
    return m.lemmatize(text)

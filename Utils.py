import re
import Settings
import random
import numpy as np
import matplotlib.pyplot as plt
from pymystem3 import Mystem
from typing import List
from nltk.tokenize import sent_tokenize
from sklearn.manifold import TSNE


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


def get_valid_text(lemmas: List[List[str]]):
    result = ""
    for sentence in lemmas:
        current_sentence = ""
        for word in sentence:
            current_sentence += word + " "
        if current_sentence != "":
            result += current_sentence + "\n"
    return result


def write_valid_text(lemmas: List[List[str]], path: str):
    # clear file
    open(path, 'w').close()
    file = open(path, 'w', encoding=Settings.PLAIN_ENCODING)
    for sentence in lemmas:
        current_sentence = ""
        for word in sentence:
            current_sentence += word + " "
        if current_sentence != "":
            file.write(current_sentence + "\n")
    return


def reduce_dimensions(vectors):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals


def plot_with_matplotlib(x_vals, y_vals, labels, seed=1):
    random.seed()

    plt.figure(figsize=Settings.FIGURE_SIZE)
    plt.scatter(x_vals, y_vals)

    # Label randomly sampled 25 data points
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, Settings.RANDOM_INDICES)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()

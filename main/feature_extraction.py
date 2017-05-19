import numpy as np
import pandas as pd
import datetime
import operator
import sklearn
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.metrics import *
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from main import function as funct


def char_len(text: str) -> int:
    return len(text)


def token_count(text: str) -> int:
    return len(nltk.word_tokenize(text))


def length_diff(text_1: str, text_2: str) -> int:
    return abs(len(text_1) - len(text_2))


def text_distance(text_1: str, text_2: str) -> float:
    tag_diff = funct.pos_tag_extraction(text_1, text_2, funct.diff_by_list)

    def avg_min_dist(tokens_1: list, tokens_2: list) -> float:
        total_dist = []
        for token_1 in tokens_1:
            token_dist = []
            for token_2 in tokens_2:
                token_dist.append(edit_distance(token_1, token_2))
            min_dist = 0 if len(token_dist) == 0 else min(token_dist)
            total_dist.append(min_dist)
        return 0 if len(tokens_1) == 0 else sum(total_dist) / len(tokens_1)

    x_tokens = [word[0] for word in tag_diff["x_diff"]]
    y_tokens = [word[0] for word in tag_diff["y_diff"]]
    return avg_min_dist(x_tokens, y_tokens) + avg_min_dist(y_tokens, x_tokens)


def word_distribution(corpus: list) -> list:
    casted_series = pd.Series(corpus).astype(str)
    words = (" ".join(casted_series)).lower().split()
    counts = Counter(words)
    result = {(word, count) for word, count in counts.items()}
    return sorted(result, key=lambda x: x[1], reverse=True)

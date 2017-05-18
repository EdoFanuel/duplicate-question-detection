import numpy as np
import pandas as pd
import datetime
import operator
import sklearn
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig


def char_len(text: str) -> int:
    return len(text)


def token_count(text: str) -> int:
    return len(nltk.word_tokenize(text))


def word_distribution(corpus: list) -> dict:
    casted_series = pd.Series(corpus).astype(str)
    words = (" ".join(casted_series)).lower().split()
    counts = Counter(words)
    return {word: count for word, count in counts.items()}

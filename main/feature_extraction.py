from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.metrics import *
from nltk.stem import WordNetLemmatizer

from main import function as func


def char_len(text: str) -> int:
    return len(text)


def token_count(text: str) -> int:
    return len(nltk.word_tokenize(text))


def length_diff(text_1: str, text_2: str) -> int:
    return abs(len(text_1) - len(text_2))


def token_distance(text_1: str, text_2: str) -> float:
    tag_diff = func.pos_tag_extraction(text_1, text_2, func.diff_by_list)

    def avg_min_dist(tokens_1: list, tokens_2: list) -> float:
        total_dist = []
        for token_1 in tokens_1:
            token_dist = []
            for token_2 in tokens_2:
                token_dist.append(levenshtein_distance(token_1, token_2))
            min_dist = 0 if len(token_dist) == 0 else min(token_dist)
            total_dist.append(min_dist)
        return 0 if len(tokens_1) == 0 else sum(total_dist) / len(tokens_1)

    x_tokens = [word[0] for word in tag_diff["x_diff"]]
    y_tokens = [word[0] for word in tag_diff["y_diff"]]
    return avg_min_dist(x_tokens, y_tokens) + avg_min_dist(y_tokens, x_tokens)


def levenshtein_distance(text_1: str, text_2: str) -> int:
    return edit_distance(text_1, text_2)


def cosine_lemma(text_1: str, text_2: str) -> float:
    pos = func.pos_tag_extraction(text_1, text_2, lambda x, y: {"pos_1": x, "pos_2": y})
    lemmatizer = WordNetLemmatizer()

    def convert(treebank_tag: str) -> str:
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    lemma_1 = [lemmatizer.lemmatize(word, pos=convert(tag)) for word, tag in pos["pos_1"]]
    lemma_2 = [lemmatizer.lemmatize(word, pos=convert(tag)) for word, tag in pos["pos_2"]]
    unique_words = set(lemma_1 + lemma_2)
    vector_1 = []
    vector_2 = []
    for word in unique_words:
        vector_1.append(int(word in lemma_1))
        vector_2.append(int(word in lemma_2))
    return func.cosine_similarity(vector_1, vector_2)


def word_distribution(corpus: list) -> list:
    casted_series = pd.Series(corpus).astype(str)
    words = (" ".join(casted_series)).lower().split()
    counts = Counter(words)
    result = {(word, count) for word, count in counts.items() if word not in stopwords.words("english")}
    return sorted(result, key=lambda x: x[1], reverse=True)

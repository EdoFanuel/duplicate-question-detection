from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.metrics import *
from nltk.stem import WordNetLemmatizer

from main import function as func

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


def word_distribution(corpus: list) -> list:
    casted_series = pd.Series(corpus).astype(str)
    words = (" ".join(casted_series)).lower().split()
    counts = Counter(words)
    result = {(word, count) for word, count in counts.items() if word not in stopwords.words("english")}
    return sorted(result, key=lambda x: x[1], reverse=True)


class FeatureExtraction:
    def __init__(self, text_1, text_2, word_dist):
        self.data_1 = FeatureExtraction.extract_basic_feature(text_1)
        self.data_2 = FeatureExtraction.extract_basic_feature(text_2)
        self.word_distribution = word_dist

    @staticmethod
    def extract_basic_feature(text: str) -> dict:
        result = {
            "content": text,
            "tokens": nltk.word_tokenize(text)
        }
        result["pos"] = nltk.pos_tag(result["tokens"])
        result["lemma"] = [lemmatizer.lemmatize(word, pos=convert(tag)) for word, tag in result["pos"]]
        return result

    def length_diff(self) -> int:
        return abs(len(self.data_1["content"]) - len(self.data_2["content"]))

    def token_distance(self) -> float:
        tag_diff = func.diff_by_list(self.data_1["pos"], self.data_2["pos"])

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

    def token_hamming(self) -> float:
        tokens = {
            "token_1": self.data_1["tokens"],
            "token_2": self.data_2["tokens"]
        }
        shrd_token = sum(1 for t1, t2 in zip(tokens["token_1"], tokens["token_2"]) if t1 == t2)
        return shrd_token / max(len(tokens["token_1"]), len(tokens["token_2"]))

    def levenshtein_distance(self) -> int:
        return edit_distance(self.data_1["content"].replace(" ", ""), self.data_2["content"].replace(" ", ""))

    def cosine_lemma(self) -> float:
        lemma_1 = self.data_1["lemma"]
        lemma_2 = self.data_2["lemma"]
        unique_words = set(lemma_1 + lemma_2)
        vector_1 = []
        vector_2 = []
        for word in unique_words:
            vector_1.append(int(word in lemma_1))
            vector_2.append(int(word in lemma_2))
        return func.cosine_similarity(vector_1, vector_2)

    def shared_token(self) -> int:
        return len(func.intersect(self.data_1["tokens"], self.data_2["tokens"]))

    def shared_pos(self) -> int:
        return len(func.intersect(self.data_1["pos"], self.data_2["pos"]))

    def shared_lemma(self) -> int:
        return len(func.intersect(self.data_1["lemma"], self.data_2["lemma"]))

    def shared_proper_noun(self) -> int:
        shrd_pos = func.intersect(self.data_1["pos"], self.data_2["pos"])
        return len([word for word, tag in shrd_pos if tag.startswith("NNP")])

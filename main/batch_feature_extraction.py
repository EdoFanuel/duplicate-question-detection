import pandas as pd
import numpy as np
import main.function as func
import main.feature_extraction as f_ext
import nltk
from nltk.corpus import stopwords
from nltk.metrics import *
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
stops = stopwords.words("english")


def extract_feature(df_in: pd.DataFrame) -> pd.DataFrame:
    df_temp = pd.DataFrame()
    df_temp["content_1"] = df_in["question1"]
    df_temp["content_2"] = df_in["question2"]
    df_temp["tokens_1"] = df_in["question1"].apply(lambda x: nltk.word_tokenize(str(x).lower()))
    df_temp["tokens_2"] = df_in["question2"].apply(lambda x: nltk.word_tokenize(str(x).lower()))
    df_temp["pos_1"] = nltk.pos_tag_sents(df_temp["tokens_1"].tolist())
    df_temp["pos_2"] = nltk.pos_tag_sents(df_temp["tokens_2"].tolist())
    df_temp["lemma_1"] = df_temp["pos_1"].apply(lambda x: (wnl.lemmatize(word, f_ext.convert(tag)) for word, tag in x))
    df_temp["lemma_2"] = df_temp["pos_1"].apply(lambda x: (wnl.lemmatize(word, f_ext.convert(tag)) for word, tag in x))

    df_out = pd.DataFrame()
    df_out["len_text_1"] = df_temp["content_1"].apply(lambda x: len(x))
    df_out["len_text_2"] = df_temp["content_2"].apply(lambda x: len(x))
    df_out["len_diff"] = np.abs(df_out["len_text_1"] - df_out["len_text_2"])
    df_out["len_char_1"] = df_temp["content_1"].apply(lambda x: len(x.replace(" ","")))
    df_out["len_char_2"] = df_temp["content_2"].apply(lambda x: len(x.replace(" ","")))
    df_out["diff_char"] = np.abs(df_out["len_char_1"] - df_out["len_char_2"])
    df_out["len_token_1"] = df_temp["tokens_1"].apply(lambda x: len(x))
    df_out["len_token_2"] = df_temp["tokens_2"].apply(lambda x: len(x))
    df_out["token_diff"] = np.abs(df_out["len_token_1"] - df_out["len_token_2"])
    df_out["shared_token"] = df_temp.apply(lambda x: len(func.intersect(x["tokens_1"], x["tokens_2"])))
    df_out["len_upper_txt_1"] = df_temp["content_1"].apply(lambda x: sum(1 for i in x if i.isupper()))
    df_out["len_upper_txt_2"] = df_temp["content_2"].apply(lambda x: sum(1 for i in x if i.isupper()))
    df_out["diff_upper"] = np.abs(df_out["len_upper_txt_1"] - df_out["len_upper_txt_2"])
    df_out["norm_levenshtein"] = df_temp.apply(lambda x: norm_levenshtein_dist(x["content_1"], x["content_2"]))
    # "len_token_1",
    # "len_token_2",
    # "shared_token",
    # "shared_token_sqrt",
    # "diff_token",
    # "token_hamming",
    # "token_dist",
    # "len_stopword_1",
    # "len_stopword_2",
    # "shared_2gram",
    # "shared_tfidf",
    # "shared_pos",
    # "shared_nnp",
    # "pos_dist",
    # "shared_lemma",
    # "cosine_lemma",
    # "avg_word_1",
    # "avg_word_2",
    # "diff_avg_word_len"
    return df_out


# Text-related features
def len_upper_txt_1(self) -> int:
    return sum([1 for i in self.data_1["content"] if i.isupper()])


def len_upper_txt_2(self) -> int:
    return sum([1 for i in self.data_2["content"] if i.isupper()])


def diff_uppercase(self) -> float:
    upper_1 = self.len_upper_txt_1()
    upper_2 = self.len_upper_txt_2()
    return abs(upper_1 / len(self.data_1["tokens"]) - upper_2 / len(self.data_2["tokens"]))


def norm_levenshtein_dist(s1: str, s2: str) -> float:
    chars_1 = s1.replace(" ", "")
    chars_2 = s2.replace(" ", "")
    dist = edit_distance(chars_1, chars_2)
    return dist / max(len(chars_1), len(chars_2))


# Token-related features
def diff_tokens(self):
    return abs(self.len_token_1() - self.len_token_2())


def token_hamming(self) -> float:
    tokens = {
        "token_1": self.data_1["tokens"],
        "token_2": self.data_2["tokens"]
    }
    shrd_token = sum(1 for t1, t2 in zip(tokens["token_1"], tokens["token_2"]) if t1 == t2)
    return shrd_token / max(self.len_token_1(), self.len_token_2())


def token_dist(self) -> float:
    tag_diff = func.diff_by_list(self.data_1["tokens"], self.data_2["tokens"])

    def avg_min_dist(tokens_1: list, tokens_2: list) -> float:
        total_dist = []
        for token_1 in tokens_1:
            token_dist = []
            for token_2 in tokens_2:
                token_dist.append(edit_distance(token_1, token_2))
            min_dist = 0 if len(token_dist) == 0 else min(token_dist)
            total_dist.append(min_dist)
        return 0 if len(tokens_1) == 0 else sum(total_dist) / len(tokens_1)

    x_tokens = [word for word in tag_diff["x_diff"]]
    y_tokens = [word for word in tag_diff["y_diff"]]
    return avg_min_dist(x_tokens, y_tokens) + avg_min_dist(y_tokens, x_tokens)


def len_stopword_txt_1(self) -> int:
    return len([token for token in self.data_1["tokens"] if token in stops])


def len_stopword_txt_2(self) -> int:
    return len([token for token in self.data_2["tokens"] if token in stops])


def diff_stopwords(self) -> float:
    return abs(self.len_stopword_txt_1() / self.len_token_1() - self.len_stopword_txt_2() / self.len_token_2())


def shared_2gram(self) -> float:
    ngram_1 = func.n_gram(self.data_1["tokens"])
    ngram_2 = func.n_gram(self.data_2["tokens"])
    if len(ngram_1) + len(ngram_2) == 0:
        return 0
    else:
        return len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2))


def shared_tf_idf(self) -> float:
    shrd_token = func.intersect(self.data_1["tokens"], self.data_2["tokens"])

    tfidf_1 = self.tfidf_model[self.dictionary.doc2bow(self.data_1["tokens"])]
    tfidf_2 = self.tfidf_model[self.dictionary.doc2bow(self.data_2["tokens"])]

    shared_dict_ids = [self.dictionary.token2id[token] for token in shrd_token if token in self.dictionary.token2id]
    shared_tfidf = [value for token_id, value in tfidf_1 + tfidf_2 if token_id in shared_dict_ids]
    total_tfidf = [value for _, value in tfidf_1 + tfidf_2]
    return sum(shared_tfidf) / sum(total_tfidf)


# Part-of-speech related features
def shared_pos(self) -> int:
    return len(func.intersect(self.data_1["pos"], self.data_2["pos"]))


def shared_proper_noun(self) -> int:
    shrd_pos = func.intersect(self.data_1["pos"], self.data_2["pos"])
    return len([word for word, tag in shrd_pos if tag.startswith("NNP")])


def pos_dist(self) -> float:
    tag_diff = func.diff_by_list(self.data_1["tokens"], self.data_2["tokens"])

    def avg_min_dist(tokens_1: list, tokens_2: list) -> float:
        total_dist = []
        for token_1 in tokens_1:
            token_dist = []
            for token_2 in tokens_2:
                token_dist.append(edit_distance(token_1, token_2))
            min_dist = 0 if len(token_dist) == 0 else min(token_dist)
            total_dist.append(min_dist)
        return 0 if len(tokens_1) == 0 else sum(total_dist) / len(tokens_1)

    x_tokens = [word for word in tag_diff["x_diff"]]
    y_tokens = [word for word in tag_diff["y_diff"]]
    return avg_min_dist(x_tokens, y_tokens) + avg_min_dist(y_tokens, x_tokens)


# Lemma related features
def shared_lemma(self) -> int:
    return len(func.intersect(self.data_1["lemma"], self.data_2["lemma"]))


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


# Combination features
def avg_word_len_txt_1(self) -> float:
    return self.len_text_1() / self.len_token_1()


def avg_word_len_txt_2(self) -> float:
    return self.len_text_2() / self.len_token_2()


def diff_avg_word_len(self) -> float:
    return abs(self.avg_word_len_txt_1() - self.avg_word_len_txt_2())

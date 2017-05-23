import math
import nltk
from gensim import corpora, models
from nltk.corpus import stopwords, wordnet
from nltk.metrics import *
from nltk.stem import WordNetLemmatizer

from main import function as func

lemmatizer = WordNetLemmatizer()
stops = stopwords.words("english")


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


class FeatureExtraction:
    def __init__(self, text_1: str, text_2: str, dictionary: corpora.Dictionary, tfidf_model: models.TfidfModel):
        self.data_1 = FeatureExtraction.extract_basic_feature(text_1)
        self.data_2 = FeatureExtraction.extract_basic_feature(text_2)
        self.dictionary = dictionary
        self.tfidf_model = tfidf_model

    @staticmethod
    def extract_basic_feature(text: str) -> dict:
        result = {
            "content": str(text),
            "tokens": func.word_tokenize(str(text).lower())
        }
        result["pos"] = nltk.pos_tag(result["tokens"])
        result["lemma"] = [lemmatizer.lemmatize(word, pos=convert(tag)) for word, tag in result["pos"]]
        return result

    def generate_features(self, data_id: int, is_duplicate: int) -> dict:
        result = {
            "id": data_id,
            "is_duplicate": is_duplicate,
            "len_text_1": self.len_text_1(),
            "len_text_2": self.len_text_2(),
            "len_diff": self.len_diff(),
            "len_char_1": self.len_char_txt_1(),
            "len_char_2": self.len_char_txt_2(),
            "diff_char": self.diff_char(),
            "len_upper_txt_1": self.len_upper_txt_1(),
            "len_upper_txt_2": self.len_upper_txt_2(),
            "diff_upper": self.diff_uppercase(),
            "norm_levenshtein": self.norm_levenshtein_dist(),
            "len_token_1": self.len_token_1(),
            "len_token_2": self.len_token_2(),
            "shared_token": self.shared_token(),
            "shared_token_sqrt": math.sqrt(self.shared_token()),
            "diff_token": self.diff_tokens(),
            "token_hamming": self.token_hamming(),
            "token_dist": self.token_dist(),
            "len_stopword_1": self.len_stopword_txt_1(),
            "len_stopword_2": self.len_stopword_txt_2(),
            "diff_stopword": self.diff_stopwords(),
            "shared_2gram": self.shared_2gram(),
            "shared_tfidf": self.shared_tf_idf(),
            "shared_pos": self.shared_pos(),
            "shared_nnp": self.shared_proper_noun(),
            "pos_dist": self.pos_dist(),
            "shared_lemma": self.shared_lemma(),
            "cosine_lemma": self.cosine_lemma(),
            "avg_word_1": self.avg_word_len_txt_1(),
            "avg_word_2": self.avg_word_len_txt_2(),
            "diff_avg_word_len": self.diff_avg_word_len()
        }
        return result

    # Text-related features
    def len_text_1(self) -> int:
        return len(self.data_1["content"])

    def len_text_2(self) -> int:
        return len(self.data_2["content"])

    def len_diff(self) -> int:
        return abs(self.len_text_1() - self.len_text_2())

    def len_char_txt_1(self) -> int:
        return len(self.data_1["content"].replace(" ", ""))

    def len_char_txt_2(self) -> int:
        return len(self.data_2["content"].replace(" ", ""))

    def diff_char(self) -> int:
        return abs(self.len_char_txt_1() - self.len_char_txt_2())

    def len_upper_txt_1(self) -> int:
        return sum([1 for i in self.data_1["content"] if i.isupper()])

    def len_upper_txt_2(self) -> int:
        return sum([1 for i in self.data_2["content"] if i.isupper()])

    def diff_uppercase(self) -> float:
        upper_1 = self.len_upper_txt_1()
        upper_2 = self.len_upper_txt_2()
        return abs(upper_1 / len(self.data_1["tokens"]) - upper_2 / len(self.data_2["tokens"]))

    def norm_levenshtein_dist(self) -> float:
        chars_1 = self.data_1["content"].replace(" ", "")
        chars_2 = self.data_2["content"].replace(" ", "")
        dist = edit_distance(chars_1, chars_2)
        return dist / max(len(chars_1), len(chars_2))

    # Token-related features
    def len_token_1(self) -> int:
        return len(self.data_1["tokens"])

    def len_token_2(self) -> int:
        return len(self.data_2["tokens"])

    def shared_token(self) -> int:
        return len(func.intersect(self.data_1["tokens"], self.data_2["tokens"]))

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

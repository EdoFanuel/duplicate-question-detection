import math
import nltk
from gensim import corpora, models


def token_extraction(text_1: str, text_2: str, extract_method: callable):
    token_1 = nltk.word_tokenize(text_1)
    token_2 = nltk.word_tokenize(text_2)
    return extract_method(token_1, token_2)


def pos_tag_extraction(text_1: str, text_2: str, extract_method: callable):
    pos_1 = nltk.pos_tag(nltk.word_tokenize(text_1))
    pos_2 = nltk.pos_tag(nltk.word_tokenize(text_2))
    return extract_method(pos_1, pos_2)


def cosine_similarity(vector_1: list, vector_2: list) -> float:
    vect_product = [a * b for a, b in zip(vector_1, vector_2)]
    sq_vect_1 = [a ** 2 for a in vector_1]
    sq_vect_2 = [a ** 2 for a in vector_2]
    return sum(vect_product) / math.sqrt(sum(sq_vect_1) * sum(sq_vect_2))


def n_gram(data: list, n: int = 2) -> set:
    return set([i for i in zip(*[data[i:] for i in range(n)])])


def length_norm(data_1, data_2):
    return math.sqrt(len(data_1) * len(data_2))


def intersect(x: list, y: list) -> list:
    return list(set(x) & set(y))


def diff(x: list, y: list) -> list:
    return [data for data in x + y if data not in intersect(x, y)]


def diff_by_list(x: list, y: list) -> dict:
    common_entry = intersect(x, y)
    return {
        "x_diff": [data for data in x if data not in common_entry],
        "y_diff": [data for data in y if data not in common_entry]
    }


def generate_tfidf(dictionary: corpora.Dictionary) -> models.TfidfModel:
    return models.TfidfModel(dictionary= dictionary)


def generate_dictionary(corpus: list, token_generator: callable = None) -> corpora.Dictionary:
    if token_generator is None:
        return corpora.Dictionary(nltk.word_tokenize(sentence) for sentence in corpus)
    else:
        return corpora.Dictionary(token_generator(corpus))


def generate_corpus_vector(text: str, dictionary: corpora.Dictionary, token_generator: callable = None) -> list:
    if token_generator is None:
        return dictionary.doc2bow(nltk.word_tokenize(text))
    else:
        return dictionary.doc2bow(token_generator(text))


def tf_idf(word, document, tf_idf) -> float:
    pass  # TODO implement

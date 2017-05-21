import math
import nltk
from main import fmgmt
from datetime import datetime as dt

inverse_index = {}


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


def term_freq(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist) -> float:
    if word not in inverse_index:
        inverse_index[word] = sum(1 for blob in bloblist if word in blob.words)
    return inverse_index[word]


def inverse_doc_freq(word, bloblist) -> float:
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist) -> float:
    return term_freq(word, blob) * inverse_doc_freq(word, bloblist)


def generate_idf(dest: str, corpus_blob):
    inverse_doc_freq = []
    start_time = dt.now()
    for sentence in corpus_blob:
        for word in sentence.words:
            print(len(inverse_index), word, n_containing(word, corpus_blob))
    for word, freq in inverse_index.items():
        inverse_doc_freq.append({
            "word": word,
            "freq": freq,
        })
    fmgmt.write_csv(inverse_doc_freq, "..\\dataset\\", dest)
    time_elapsed = dt.now() - start_time
    print("IDF generated in {0} seconds".format(time_elapsed.total_seconds()))

import nltk


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
    return sum(vect_product) / (sum(sq_vect_1) * sum(sq_vect_2))


def intersect(x: list, y: list) -> list:
    return list(set(x) & set(y))


def diff(x: list, y: list) -> list:
    return [data for data in x + y if data not in intersect(x, y)]


def diff_by_list(x: list, y: list) -> dict:
    return {
        "x_diff": [data for data in x if data not in intersect(x, y)],
        "y_diff": [data for data in y if data not in intersect(x, y)]
    }

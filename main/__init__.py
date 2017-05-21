from datetime import datetime as dt
from textblob import TextBlob as tb
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

if __name__ == '__main__':
    path = "..\\dataset\\train.csv"
    start_index = 0
    limit = 100
    train_set = fmgmt.read_csv(path)[start_index: start_index + limit]

    result = []
    corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
    corpus_blob = [tb(sentence) for sentence in corpus]

    i = 0
    train_tuple = zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"])
    for q1, q2, is_duplicate in train_tuple:
        feature = f_ext.FeatureExtraction(q1, q2, corpus_blob)
        i += 1
        data = {
            "no": i,
            "q1": q1,
            "q2": q2,
            "is_duplicate": is_duplicate,
            "shared_token": feature.shared_token(),
            "shared_pos": feature.shared_pos(),
            "shared_lemma": feature.shared_lemma(),
            "shared_proper_noun": feature.shared_proper_noun(),
            "diff_uppercase": feature.diff_uppercase(),
            "diff_stopwords": feature.diff_stopwords(),
            "diff_token": feature.diff_tokens(),
            "token_distance": feature.token_distance(),
            "char_length_diff": feature.length_diff(),
            "leven_dist": feature.norm_levenshtein_distance(),
            "cosine_lemma": feature.cosine_lemma(),
            "token_hamming": feature.token_hamming(),
            "shared_2gram": feature.shared_2gram(),
            "shared_tf_idf": feature.shared_tf_idf()
        }
        result.append(data)
        print("Progress: {0} / {1}".format(i, len(train_set)))

    project_root = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\duplicate-question-detection\\"
    file_name = "result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))
    columns = [
        "no",
        "q1",
        "q2",
        "is_duplicate",
        "shared_token",
        "shared_pos",
        "shared_lemma",
        "shared_proper_noun",
        "diff_uppercase",
        "diff_stopwords",
        "diff_token",
        "token_distance",
        "char_length_diff",
        "leven_dist",
        "cosine_lemma",
        "token_hamming",
        "shared_2gram",
        "shared_tf_idf"
    ]
    fmgmt.write_csv(result, project_root, file_name, columns=columns)

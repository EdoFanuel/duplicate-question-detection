from datetime import datetime as dt
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

if __name__ == '__main__':
    path = "..\\dataset\\train.csv"
    start_index = 0
    limit = 100
    train_set = fmgmt.read_csv(path)[start_index: start_index + limit]

    result = []
    word_dist = f_ext.word_distribution(train_set["question1"].tolist() + train_set["question2"].tolist())
    train_tuple = zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"])
    i = 0
    for q1, q2, is_duplicate in train_tuple:
        feature = f_ext.FeatureExtraction(q1, q2, word_dist)
        i += 1
        data = {
            "id": i,
            "q1": q1,
            "q2": q2,
            "is_duplicate": is_duplicate,
            "shared_token": feature.shared_token(),
            "shared_pos": feature.shared_pos(),
            "shared_lemma": feature.shared_lemma(),
            "shared_proper_noun": feature.shared_proper_noun(),
            "token_distance": feature.token_distance(),
            "char_length_diff": feature.length_diff(),
            "leven_dist": feature.levenshtein_distance(),
            "cosine_lemma": feature.cosine_lemma(),
            "token_hamming": feature.token_hamming()
        }
        result.append(data)
        print("Progress: {0} / {1}".format(i, len(train_set)))

    project_root = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\duplicate-question-detection\\"
    file_name = "result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))
    columns = [
        "id",
        "q1",
        "q2",
        "is_duplicate",
        "shared_token",
        "shared_pos",
        "shared_lemma",
        "shared_proper_noun",
        "token_distance",
        "char_length_diff",
        "leven_dist",
        "cosine_lemma",
        "token_hamming"
    ]
    fmgmt.write_csv(result, project_root, file_name, columns=columns)

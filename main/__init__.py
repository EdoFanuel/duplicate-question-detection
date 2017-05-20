from datetime import datetime as dt
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

if __name__ == '__main__':
    path = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\train.csv"
    start_index = 0
    limit = 20
    train_set = fmgmt.read_csv(path)[start_index: start_index + limit]

    result = []
    # word_dist = f_ext.word_distribution(train_set["question1"].tolist() + train_set["question2"].tolist())
    train_tuple = zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"])
    i = 0
    for q1, q2, is_duplicate in train_tuple:
        data = {
            "q1": q1,
            "q2": q2,
            "is_duplicate": is_duplicate,
            "shared_token": f_ext.shared_token(q1, q2),
            "shared_pos": f_ext.shared_pos(q1, q2),
            "shared_lemma": f_ext.shared_lemma(q1, q2),
            "shared_proper_noun": f_ext.shared_proper_noun(q1, q2),
            "token_distance": f_ext.token_distance(q1, q2),
            "char_length_diff": f_ext.length_diff(q1, q2),
            "leven_dist": f_ext.levenshtein_distance(q1.replace(" ", ""), q2.replace(" ", "")),
            "cosine_lemma": f_ext.cosine_lemma(q1, q2)
        }
        result.append(data)
        i += 1
        print("Progress: {0} / {1}".format(i, len(train_set)))
    project_root = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\duplicate-question-detection\\"
    fmgmt.write_csv(result, project_root, "result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S")))

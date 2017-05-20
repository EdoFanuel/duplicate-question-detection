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
    word_dist = f_ext.word_distribution(train_set["question1"].tolist() + train_set["question2"].tolist())
    for q1, q2, is_duplicate in zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"]):
        data = {
            "q1": q1,
            "q2": q2,
            "is_duplicate": is_duplicate,
            "token_distance": f_ext.token_distance(q1, q2),
            "char_length_diff": f_ext.length_diff(q1, q2),
            "leven_dist": f_ext.edit_distance(q1.replace(" ", ""), q2.replace(" ", ""))
        }
        result.append(data)
    fmgmt.write_csv(result, result[0].keys(), "../result/{0}.csv".format(str(dt.now())))

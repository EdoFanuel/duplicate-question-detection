from datetime import datetime as dt
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct
import os.path as osp

if __name__ == '__main__':
    train_file = "..\\dataset\\train.csv"
    test_file = "..\\dataset\\test.csv"
    dict_file = "..\\feature\\quora_trainset.dict"
    train_set = fmgmt.read_csv(train_file)

    result = []
    print("Start collecting corpus blob for {0} documents".format(len(train_set)))
    start_time = dt.now()
    corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
    time_elapsed = dt.now() - start_time
    # Read and fill inverse-index if file is already generated
    dictionary = fmgmt.reload_dictionary(dict_file, corpus)
    tfidf_model = funct.generate_tfidf(dictionary)

    i = 0
    train_tuple = zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"])
    for q1, q2, is_duplicate in train_tuple:
        feature = f_ext.FeatureExtraction(q1, q2, dictionary, tfidf_model)
        i += 1
        data = feature.generate_features(i, is_duplicate)
        result.append(data)
        print("Progress: {0} / {1}".format(i, len(train_set)))

    project_root = "..\\"
    file_name = "result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))

    fmgmt.write_csv(result, project_root, file_name)

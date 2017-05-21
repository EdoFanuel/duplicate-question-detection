from datetime import datetime as dt
from textblob import TextBlob as tb
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct
import os.path as osp

if __name__ == '__main__':
    train_file = "..\\dataset\\train.csv"
    test_file = "..\\dataset\\test.csv"
    idf_file = "..\\dataset\\inverse_doc_freq_test.csv"
    train_set = fmgmt.read_csv(train_file)

    result = []
    print("Start collecting corpus blob for {0} documents".format(len(train_set)))
    start_time = dt.now()
    corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
    corpus_blob = [tb(sentence) for sentence in corpus if isinstance(sentence, str)]
    time_elapsed = dt.now() - start_time
    print("Finished collecting corpus blob in {0} seconds".format(time_elapsed.total_seconds()))
    # Read and fill inverse-index if file is already generated
    if osp.isfile(idf_file):
        idf_set = fmgmt.read_csv(idf_file)
        for word, doc_freq in zip(idf_set["word"], idf_set["freq"]):
            funct.inverse_index[word] = doc_freq
    else:
        funct.generate_idf(idf_file, corpus_blob)

    i = 0
    train_tuple = zip(train_set["question1"], train_set["question2"], train_set["is_duplicate"])
    for q1, q2, is_duplicate in train_tuple:
        feature = f_ext.FeatureExtraction(q1, q2, corpus_blob)
        i += 1
        data = feature.generate_features(i, is_duplicate)
        result.append(data)
        print("Progress: {0} / {1}".format(i, len(train_set)))

    project_root = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\duplicate-question-detection\\"
    file_name = "result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))

    fmgmt.write_csv(result, project_root, file_name)

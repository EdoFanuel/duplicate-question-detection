from datetime import datetime as dt
import os.path as osp

import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def generate_feature(data_file: str, dict_file: str, feature_file: str) -> pd.DataFrame:
    train_set = fmgmt.read_csv(data_file, {"question1": str, "question2": str, "is_duplicate": int})

    result = []
    print("Start collecting corpus for {0} documents".format(len(train_set)))
    corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
    # Read and fill inverse-index if file is already generated
    dictionary = fmgmt.reload_dictionary(dict_file, corpus)
    tfidf_model = funct.generate_tfidf(dictionary)

    i = 0
    for train_data in train_set.iterrows():
        q1, q2, is_duplicate, test_id = train_data["question1"], train_data["question2"], train_data["is_duplicate"], train_data["id"]
        feature = f_ext.FeatureExtraction(q1, q2, dictionary, tfidf_model)
        i += 1
        data = feature.generate_features(test_id, is_duplicate)
        result.append(data)
        if i % 100 == 0:
            print("Progress: {0} / {1}".format(i, len(train_set)))
    print("Saving features as {0}".format(feature_file))
    return fmgmt.write_csv(result, feature_file)


if __name__ == '__main__':
    train_file = "..\\dataset\\train.csv"
    train_dict_file = "..\\feature\\quora_trainset.dict"
    train_feature_file = "..\\feature\\train_feature.csv"

    test_file = "..\\dataset\\test.csv"
    test_dict_file = "..\\feature\\quora_testset.dict"
    test_feature_file = "..\\feature\test_feature.csv"
    file_name = "..\\result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))

    # Generating training features
    if osp.isfile(train_feature_file):
        train_feature = fmgmt.read_csv(train_feature_file)
    else:
        train_feature = generate_feature(train_file, train_dict_file, train_feature_file)

    # Initialize random forest
    clf = RandomForestClassifier()
    clf.fit(train_feature[f_ext.FeatureExtraction.get_feature_fields()], train_feature["is_duplicate"])

    # Generating test features
    if osp.isfile(test_feature_file):
        test_feature = fmgmt.read_csv(test_feature_file)
    else:
        test_feature = generate_feature(test_file, test_dict_file, test_feature_file)

    # Find result
    prediction = clf.predict_proba(test_feature[f_ext.FeatureExtraction.get_feature_fields()])
    final_result = []
    for row in test_feature.iterrows():
        final_result.append({
            "test_id": row["id"],
            "is_duplicate": prediction[row["id"]][1]
        })

    # Save result
    fmgmt.write_csv(final_result, file_name)

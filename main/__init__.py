from datetime import datetime as dt
import os.path as osp

import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct
import main.script as script

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    train_file = "..\\dataset\\train.csv"
    train_dict_file = "..\\feature\\quora_trainset.dict"
    train_feature_file = "..\\feature\\train_feature.csv"

    test_file = "..\\dataset\\test.csv"
    test_dict_file = "..\\feature\\quora_testset.dict"
    test_feature_file = "..\\feature\\test_feature.csv"
    file_name = "..\\result\\{0}.csv".format(dt.now().strftime("%Y-%m-%d %H-%M-%S"))

    # Generating training features
    print("Loading training features...")
    if osp.isfile(train_feature_file):
        train_feature = fmgmt.read_csv(train_feature_file)
    else:
        train_feature = script.generate_feature(train_file, train_dict_file, train_feature_file, training_mode=True)

    # Initialize random forest
    features = ["cosine_lemma", "norm_levenshtein", "shared_2gram", "shared_tfidf", "shared_token"]
    print("Loading Random Forest...")
    clf = RandomForestClassifier()
    clf.fit(train_feature[features], train_feature["is_duplicate"])

    # Generating test features
    print("Loading test features...")
    if osp.isfile(test_feature_file):
        test_feature = fmgmt.read_csv(test_feature_file)
    else:
        test_feature = script.generate_feature(test_file, test_dict_file, test_feature_file, training_mode=False)

    # Find result
    print("Predicting values...")
    predictions = clf.predict_proba(test_feature[features])
    print("Saving result")
    final_result = pd.DataFrame()
    final_result["test_id"] = test_feature["id"]
    final_result["is_duplicate"] = [value[1] for value in predictions]

    # Save result
    final_result.to_csv(file_name, index=False)
    print("Important features:")
    for feature, importance in sorted(list(zip(features, clf.feature_importances_)), key=lambda x: x[1], reverse=True):
        print(feature, importance)

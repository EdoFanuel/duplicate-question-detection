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
    if osp.isfile(train_feature_file):
        train_feature = fmgmt.read_csv(train_feature_file)
    else:
        train_feature = script.generate_feature(train_file, train_dict_file, train_feature_file, training_mode=True)

    # Initialize random forest
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(train_feature[f_ext.FeatureExtraction.get_feature_fields()], train_feature["is_duplicate"])

    # Generating test features
    if osp.isfile(test_feature_file):
        test_feature = fmgmt.read_csv(test_feature_file)
    else:
        test_feature = script.generate_feature(test_file, test_dict_file, test_feature_file, training_mode=False)

    # Find result
    print("Predicting values...")
    preds = clf.predict_proba(test_feature[f_ext.FeatureExtraction.get_feature_fields()])
    print("Saving result")
    final_result = pd.DataFrame()
    final_result["test_id"] = test_feature["id"]
    final_result["is_duplicate"] = [pred[1] for pred in preds]

    # Save result
    final_result.to_csv(file_name, index=False)

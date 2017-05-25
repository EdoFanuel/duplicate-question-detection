import main.function as funct
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import pandas as pd


def generate_feature(data_file: str, dict_file: str, feature_file: str, training_mode: bool = True, start_index: int = None, end_index: int = None) -> pd.DataFrame:
    train_set = fmgmt.read_csv(data_file, {"question1": str, "question2": str, "is_duplicate": int})

    result = []
    print("Start collecting corpus for {0} documents".format(len(train_set)))
    corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
    # Read and fill inverse-index if file is already generated
    dictionary = fmgmt.reload_dictionary(dict_file, corpus)
    tfidf_model = funct.generate_tfidf(dictionary)

    i = 0
    if start_index is not None and end_index is not None:
        train_set = pd.DataFrame(train_set[start_index: min(end_index, len(train_set))])

    for index, train_data in train_set.iterrows():
        q1, q2 = train_data["question1"], train_data["question2"]
        feature = f_ext.FeatureExtraction(q1, q2, dictionary, tfidf_model)
        i += 1
        data = feature.generate_features()
        if training_mode:
            data["is_duplicate"] = train_data["is_duplicate"]
            data["id"] = train_data["id"]
        else:
            data["id"] = train_data["test_id"]
        result.append(data)
        if i % 100 == 0:
            print("\tProgress: {0} / {1}".format(i, len(train_set)))
    print("Saving features as {0}".format(feature_file))
    return fmgmt.write_csv(result, feature_file)

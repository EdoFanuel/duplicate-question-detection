from main import script
import pandas as pd
import math
import os.path as osp

test_file = "..\\dataset\\test.csv"
test_feature_file = "..\\feature\\test_feature.csv"
test_dict_file = "..\\feature\\quora_testset.dict"

entries_per_file = 10000
total_file = round(math.ceil(2345796 / entries_per_file))
partial_data = []
for i in range(0, total_file):
    print("Starting Batch {0} / {1}".format(i + 1, total_file))
    partial_file = "..\\feature-partial\\test_feature_p{0}.csv".format(i + 1)
    if osp.isfile(partial_file):
        partial_data.append(pd.read_csv(partial_file))
    else:
        test_feature = script.generate_feature(
            data_file=test_file,
            dict_file=test_dict_file,
            feature_file=partial_file,
            training_mode=False,
            start_index=i * entries_per_file,
            end_index=(i+1) * entries_per_file
        )
        partial_data.append(test_feature)
print("Feature extracted. Saving result on {0}".format(test_feature_file))
total_data = pd.concat(partial_data)
total_data.sort_values("id")
total_data.to_csv(test_feature_file, index=False, encoding="utf-8")
print("Done")

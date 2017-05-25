from main import script
import math
import os.path as osp

test_file = "..\\dataset\\test.csv"
test_dict_file = "..\\feature\\quora_testset.dict"

entries_per_file = 10000
total_file = round(math.ceil(2345796 / entries_per_file))
for i in range(0, total_file):
    print("Starting Batch {0} / {1}".format(i + 1, total_file))
    test_feature_file = "..\\feature\\test_feature_p{0}.csv".format(i + 1)
    if osp.isfile(test_feature_file):
        continue
    else:
        test_feature = script.generate_feature(
            data_file=test_file,
            dict_file=test_dict_file,
            feature_file=test_feature_file,
            training_mode=False,
            start_index=i * entries_per_file,
            end_index=(i+1) * entries_per_file
        )

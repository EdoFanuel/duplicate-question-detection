from datetime import datetime as dt
import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

path = "..\\dataset\\train.csv"
start_index = 0
limit = 100
train_set = fmgmt.read_csv(path)


def execution_time(sentences: list, executed_func: callable) -> (int, list):
    start_time = dt.now()
    result = executed_func(sentences)
    end_time = dt.now()
    time_elapsed = end_time - start_time
    return time_elapsed.total_seconds(), result


corpus = train_set["question1"].tolist() + train_set["question2"].tolist()
print("word_dist: {1} entries in {0} seconds".format(execution_time(corpus, f_ext.word_distribution)[0], len(corpus)))
print("token_dist: {1} entries in {0} seconds".format(execution_time(corpus, f_ext.token_distribution)[0], len(corpus)))

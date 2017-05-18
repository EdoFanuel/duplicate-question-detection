import main.file_management as fmgmt
import main.feature_extraction as f_ext

if __name__ == '__main__':
    path = input("Training dataset path: ")
    train_set = fmgmt.read_csv(path)
    word_dist = f_ext.word_distribution(train_set["question1"].tolist() + train_set["question2"].tolist())
    print(word_dist)


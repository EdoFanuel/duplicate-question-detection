import main.file_management as fmgmt
import main.feature_extraction as f_ext
import main.function as funct

if __name__ == '__main__':
    path = "D:\\Miscellanous\\Competitions\\Quora Question Pairs\\train.csv"
    train_set = fmgmt.read_csv(path)
    for q1, q2, is_duplicate in zip(train_set["question1"][:20], train_set["question2"][:20], train_set["is_duplicate"][:20]):
        print(is_duplicate, f_ext.text_distance(q1, q2))
    # word_dist = f_ext.word_distribution(train_set["question1"].tolist() + train_set["question2"].tolist())
    # for word in word_dist[:20]:
    #     print(word[0], word[1])

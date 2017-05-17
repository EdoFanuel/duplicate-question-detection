import main.file_management as fmgmt
import main.feature_extraction as f_ext

if __name__ == '__main__':
    path = input("Training dataset path: ")
    train_set = fmgmt.read_csv(path)
    for text in train_set['question1'][:20]:
        print(f_ext.word_tokenize(text), text)


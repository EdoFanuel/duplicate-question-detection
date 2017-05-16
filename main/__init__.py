import nltk
from main import function

if __name__ == '__main__':
    trainingSet = function.readCSV("D:/Miscellanous/Competitions/Quora Question Pairs/train.csv");
    for line in trainingSet:
        print(line)

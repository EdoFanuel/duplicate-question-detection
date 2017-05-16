import nltk
import fileinput
from main import function

if __name__ == '__main__':
    path = input("Training dataset path: ")
    trainingSet = function.readCSV(path);
    for line in trainingSet:
        print(line)

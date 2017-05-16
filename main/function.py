import csv
import nltk

def readCSV(srcPath: str) -> list:
    if type(srcPath) is not str:
        raise TypeError
    reader = csv.reader(open(srcPath, encoding="UTF-8"))
    content = list(reader)
    title = content[0]
    result = []
    for line in content[1:]:
        entry = {}
        for index, data in enumerate(line):
            entry[title[index]] = data
        result.append(entry)
    return result

def writeCSV(headings: list, content: list, destPath: str):
    if type(content) is not list or type(headings) is not list or type(destPath) is not str:
        raise TypeError
    else:
        writer = csv.writer(open(destPath, encoding="UTF-8"))
        writer.writerow(headings)
        pass #TODO write contents

def similarity(text_01: str, text_02: str) -> float:
    if type(text_01) is not str or type(text_02) is not str:
        raise TypeError
    else:
        pass #TODO implement
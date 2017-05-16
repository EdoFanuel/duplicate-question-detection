import csv

def readCSV(srcPath) -> list:
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

def writeCSV(headings, content, destPath):
    writer = csv.writer(open(destPath, encoding="UTF-8"))

def similarity(text_01, text_02) -> float:
    pass #TODO implement
import csv
import hashlib
import string
import random
from os import listdir
from os.path import isfile, join

class Utility:
    newLine = '\r\n'
    delimiter = ','
    quotechar = '|'
    commentHeader = '#'

    @staticmethod
    def getFiles(dirPath):
        return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

    @staticmethod
    def readCsv(filePath):
        dataList = []
        with open(filePath, newline=Utility.newLine) as fileObj:
            reader = csv.reader(fileObj, delimiter=Utility.delimiter, quotechar=Utility.quotechar)

            stripFunc = lambda i: i.strip().replace(' ', '')
            for data in reader:
                if (not data[0].startswith(Utility.commentHeader, 0)):
                    newData = list(map(stripFunc, data))
                    dataList.append(newData)

        return dataList

    @staticmethod
    def writeCsv(filePath, dataList):
        with open(filePath, 'w', newline='') as fileObj:
            writer = csv.writer(fileObj, delimiter=Utility.delimiter, quotechar=Utility.quotechar, quoting=csv.QUOTE_MINIMAL)
            for data in dataList:
                writer.writerow(data)

    @staticmethod
    def generateHash(data):
        # Assumes the default UTF-8
        hash_object = hashlib.md5(data.encode())
        return hash_object.hexdigest()

    @staticmethod
    def generateHashFromFile(filePath):

        dataList = Utility.readCsv(filePath)

        strippedData = ''.join(list(map(lambda littleList: ''.join(list(map(lambda i: i.strip().replace(' ', ''), littleList))), dataList)))

        return Utility.generateHash(strippedData)

    @staticmethod
    def generateRandom(stringLength=8):
        letters = string.ascii_lowercase+string.digits
        return ''.join(random.choice(letters) for i in range(stringLength))

    @staticmethod
    def sortBy(dataList, sortFunc):
        # L = [["Alice", 20.233], ["Bob", 20.454], ["Alex", 5.978]]
        dataList.sort(key=sortFunc, reverse=True)

        return dataList

    @staticmethod
    def generateRandomFloat(min, max):
        multiplier = float(1.0/min)
        minInt = multiplier*min
        maxInt = multiplier*max
        return random.randrange(minInt, maxInt+1)
    
    @staticmethod
    def getValue(value, typeStr):
        if (value == ''):
            return None
        
        if (typeStr == 'int'):
            return int(value)
        elif (typeStr == 'float'):
            return float(value)
        else:
            return value
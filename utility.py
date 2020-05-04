import csv
import hashlib
import string
import random
from os import listdir
from os.path import isfile, join

class Utility:
    newLine = '\n'
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
    def writeList(filePath, dataList):
        filePtr = open(filePath, "w")
        for line in dataList:
            # write line to output file
            filePtr.write(line+'\n')

        filePtr.close()

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
    def generateRandom(stringLength=10):
        letters = string.ascii_lowercase+string.digits
        return ''.join(random.choice(letters) for i in range(stringLength))

    @staticmethod
    def generateRandomFloat(min, max):
        multiplier = float(1.0/min)
        minInt = multiplier*min
        maxInt = multiplier*max
        return random.randrange(minInt, maxInt+1)/multiplier
    
    @staticmethod
    def getRandom(list):
        return random.choice(list)

    @staticmethod
    def randomInt(ranges, rangeName):
        found = list(filter(lambda x: x[0] == rangeName, ranges))[0]
        min = int(found[1])
        max = int(found[2])
        return random.randint(min, max)

    @staticmethod
    def randomFloat(ranges, rangeName):
        found = list(filter(lambda x: x[0] == rangeName, ranges))[0]
        min = float(found[1])
        max = float(found[2])
        return Utility.generateRandomFloat(min, max)

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

from utility import Utility
import hashlib

filePath = './models/model.csv'
newFilePath = './models/new.csv'

dataList = Utility.readCsv(filePath)

Utility.writeCsv(newFilePath, dataList)

print(Utility.generateRandom())

hash = Utility.generateHashFromFile(newFilePath)

for i in range(100):
    print(Utility.generateRandom())
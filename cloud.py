# $Env:Path
# For Anaconda, run:
# conda init powershell
# Comment it out if not connecting to Google Drive
# Please run pip install google.colab
from google.colab import drive
import csv

class Cloud:
    @staticmethod
    def connectToCloudDrive():
        drive.mount('/content/drive', force_remount=True)

    @staticmethod
    def disconnectFromCloudDrive():
        # drive.flush_and_unmount()
        print('drive.flush_and_unmount()')

    @staticmethod
    def getCloudDriveBase():
        return '/content/drive/My Drive/Colab Notebooks/'

    @staticmethod
    def testCsv(csvPath):
        with open(csvPath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                print(', '.join(row))

    @staticmethod
    def testCsvOnCloudDrive(filename):
        Cloud.connectToCloudDrive()
        filePath = Cloud.getCloudDriveBase() + filename
        Cloud.testCsv(filePath)
        Cloud.disconnectFromCloudDrive()
    


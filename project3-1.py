from __future__ import print_function

# %tensorflow_version 2.x

import csv
import timeit
import copy
import shutil

import tensorflow as tf

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers

# Importing the EMNIST letters
from scipy import io as sio

# from cloud import Cloud
from hardware import Hardware
from utility import Utility

class Data2d:
    def __init__(self, xData, yData):
        self.x = xData
        self.y = yData

class InputShape:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.modified = None

    def normalized(self):
        return self.modified

class LayerDescriptor:
    def __init__(self, modelDescriptor, inputList, index, descriptionCount):
        self.modelDescriptor = modelDescriptor
        self.InputList = inputList
        self.LayerType = inputList[0]
        self.data = ','.join(inputList)
        self.index = index
        self.descriptionCount = descriptionCount
        self.Activation = None
        self.FilterCount = None
        self.KernelSize = None

    def build(self):
        layerType = self.LayerType
        inputList = self.InputList

        if (layerType == 'conv2d'):
            self.Activation = inputList[1]
            self.FilterCount = int(inputList[2])
            self.KernelSize = (int(inputList[3]), int(inputList[4]))

            kernelRegularizer = LayerDescriptor.kernelRegularizer(inputList[3], inputList[4])
            if (self.index == 0):
                layer = Conv2D(self.FilterCount, kernel_size=self.KernelSize, activation=self.Activation, kernel_regularizer=kernelRegularizer, input_shape=self.modelDescriptor.inputShape.normalized())
            else:
                layer = Conv2D(self.FilterCount, kernel_size=self.KernelSize, activation=self.Activation, kernel_regularizer=kernelRegularizer)

        elif (self.LayerType == 'maxpooling2d'):
            layer = MaxPooling2D(pool_size=(int(inputList[1]), int(inputList[2])))

        elif (self.LayerType == 'dropout'):
            layer = Dropout(float(inputList[1]))

        elif (self.LayerType == 'flatten'):
            layer = Flatten()

        elif (self.LayerType == 'dense'):
            self.Activation = inputList[1]

            if (self.index+1 == self.descriptionCount):
                layer = Dense(self.modelDescriptor.numberOfClasses, activation=self.Activation)
            else:
                layer = Dense(int(inputList[2]), activation=self.Activation)
        
        return layer

    @staticmethod
    def kernelRegularizer(funcName, value):

        if (funcName == 'l1'):
            return regularizers.l1(value)
        elif (funcName == 'l2'):
            return regularizers.l2(value)
        else:
            return None

    @staticmethod
    def create(modelDescriptor, inputList, index, descriptionCount):
        return LayerDescriptor(modelDescriptor, inputList, index, descriptionCount)

class LayerInstance:
    def __init__(self, layer, descriptor):
        self.layer = layer
        self.descriptor = descriptor

class ModelData:
    def __init__(self, train, test, val, inputShape):
        self.train = train
        self.test = test
        self.val = val
        self.inputShape = inputShape

class ModelDescriptor:
    def __init__(self, descriptorFilepath, archivePath, badPath, hyperparameters, descriptions, inputShape, numberOfClasses, threshholdAccuracy):
        self.descriptorFilepath = descriptorFilepath
        self.archivePath = archivePath
        self.badPath = badPath
        self.hyperparameters = hyperparameters
        self.descriptions = descriptions
        self.threshholdAccuracy = threshholdAccuracy

        # processCount, epochCount, batchSize, kernelRegularizer, kernelRegularizerValue, loss, optimizer
        i = -1

        i += 1
        self.processCount = int(hyperparameters[i])
        i += 1
        self.epochCount = int(hyperparameters[i])
        i += 1
        self.batchSize = int(hyperparameters[i])
        i += 1
        self.loss = hyperparameters[i]
        i += 1
        self.optimizer = hyperparameters[i]

        self.inputShape = inputShape
        self.numberOfClasses = numberOfClasses
        self.hash = Utility.generateHashFromFile(descriptorFilepath)
    
    def archive(self):
        shutil.move(self.descriptorFilepath,  self.archivePath)

    def bad(self):
        shutil.move(self.descriptorFilepath,  self.badPath)

    @staticmethod
    def create(descriptorFilepath, archivePath, badPath, inputShape, numberOfClasses, threshholdAccuracy):
        descriptions = Utility.readCsv(descriptorFilepath)
        hyperparameters = descriptions.pop(0)
        return ModelDescriptor(descriptorFilepath, archivePath, badPath, hyperparameters, descriptions, inputShape, numberOfClasses, threshholdAccuracy)

class ModelInstance:
    def __init__(self, descriptor, model=None, layers=None, score=None):
        self.descriptor = descriptor
        self.model = model
        self.layers = layers
        self.score = score

    @staticmethod
    def loadModelData(matFilepath, modelDescriptor):

        inputShape = copy.deepcopy(modelDescriptor.inputShape)
        numberOfClasses = modelDescriptor.numberOfClasses

        # https://stackoverflow.com/questions/51125969/loading-emnist-letters-dataset/53547262#53547262
        # mat = sio.loadmat('emnist-letters.mat')
        mat = sio.loadmat(matFilepath)
        data = mat['dataset']

        train = Data2d(data['train'][0,0]['images'][0,0], data['train'][0,0]['labels'][0,0])
        test = Data2d(data['test'][0,0]['images'][0,0], data['test'][0,0]['labels'][0,0])

        trainShape = train.x.shape[0]
        testShape = test.x.shape[0]

        val_start = trainShape - testShape
        val = Data2d(train.x[val_start:trainShape,:], train.y[val_start:trainShape])
        train.x = train.x[0:val_start,:]
        train.y = train.y[0:val_start]

        trainShape = train.x.shape[0]
        testShape = test.x.shape[0]
        valShape = val.x.shape[0]

        if K.image_data_format() == 'channels_first':
            train.x = train.x.reshape(trainShape, 1, inputShape.rows, inputShape.cols)
            test.x = test.x.reshape(testShape, 1, inputShape.rows, inputShape.cols)
            val.x = val.x.reshape(valShape, 1, inputShape.rows, inputShape.cols)
            inputShape.modified = (1, inputShape.rows, inputShape.cols)
        else:
            train.x = train.x.reshape(trainShape, inputShape.rows, inputShape.cols, 1)
            test.x = test.x.reshape(testShape, inputShape.rows, inputShape.cols, 1)
            val.x = val.x.reshape(valShape, inputShape.rows, inputShape.cols, 1)
            inputShape.modified = (inputShape.rows, inputShape.cols, 1)

        train.x = train.x.astype('float32')
        test.x = test.x.astype('float32')
        train.x /= 255
        test.x /= 255
        print('x_train shape:', train.x.shape)
        print(train.x.shape[0], 'train samples')
        print(test.x.shape[0], 'test samples')


        # convert class vectors to binary class matrices
        train.y = tf.keras.utils.to_categorical(train.y - 1, numberOfClasses, dtype='float32')
        test.y = tf.keras.utils.to_categorical(test.y - 1, numberOfClasses, dtype='float32')

        val.y = tf.keras.utils.to_categorical(val.y - 1, numberOfClasses, dtype='float32')

        return ModelData(train, test, val, inputShape)

    @staticmethod
    def buildLayers(modelInstance, descriptions):
        layers = []

        descriptionCount = len(descriptions)
        for i in range(descriptionCount):
            desc = descriptions[i]
            layerDescriptor = LayerDescriptor.create(modelInstance.descriptor, desc, i, descriptionCount)
            layers.append(LayerInstance(layerDescriptor.build(), layerDescriptor))

        modelInstance.layers = layers

        return modelInstance

    @staticmethod
    def buildModel(modelInstance, layerInstances):
        model = Sequential()

        for i in range(len(layerInstances)):
            layer = layerInstances[i].layer
            layerDescriptor = layerInstances[i].descriptor
            print("i: %d type: %s data: %s"%(i, layerDescriptor.LayerType, layerDescriptor.data)) 
            model.add(layer)
            print("i: %d added"%(i)) 

        model.compile(loss=modelInstance.descriptor.loss,
                    optimizer=modelInstance.descriptor.optimizer,
                    metrics=['accuracy'])
        
        modelInstance.model = model

        return modelInstance

    @staticmethod
    def create(descriptor):
        return ModelInstance(descriptor)

class ScoreBoard:
    def __init__(self, scoreFilepath):
        self.scoreFilepath = scoreFilepath
        self.scoreList = Utility.readCsv(scoreFilepath)
    
    def isDone(self, modelDescriptor):
        itemList = self.getItems(modelDescriptor.hash)
        if (len(itemList) > 0):
            scoreObj = itemList[0]
            
            accuracy = float(scoreObj[1])
            return ((accuracy < modelDescriptor.threshholdAccuracy) or (int(scoreObj[4]) >= modelDescriptor.processCount))

        return False

    def isInList(self, identifier):
        return True in (i[0] == identifier for i in self.scoreList)
    
    def getItems(self, identifier):
        return list(filter(lambda x: x[0] == identifier, self.scoreList))

    def record(self, modelDescriptor, score):

        scoreObj = None
        count = 0
        itemList = self.getItems(modelDescriptor.hash)
        if (len(itemList) > 0):
            scoreObj = itemList[0]
            count = int(scoreObj[4]) + 1
            scoreObj[4] = count
            if (score[1] > float(scoreObj[1])):
                scoreObj[1] = score[1]
                scoreObj[2] = score[0]
        else:
            count = 1
            scoreObj = [modelDescriptor.hash, score[1], score[0], modelDescriptor.descriptorFilepath, count]
            self.scoreList.append(scoreObj)
        
        self.scoreList.sort(key=lambda i:float(i[1]), reverse=True)

        newList = copy.copy(self.scoreList)
        newList.insert(0, ['#hash', 'score1', 'score0', 'filePath', 'count'])
        Utility.writeCsv(self.scoreFilepath, newList)

        if (self.isDone(modelDescriptor)):
            modelDescriptor.archive()

def processNN(matFilepath, modelInstance):
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    earlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
        )

    modelData = ModelInstance.loadModelData(matFilepath, modelInstance.descriptor)

    modelInstance.descriptor.inputShape = modelData.inputShape

    modelInstance = ModelInstance.buildLayers(modelInstance, modelInstance.descriptor.descriptions)

    try:
        modelInstance = ModelInstance.buildModel(modelInstance, modelInstance.layers)

        history = modelInstance.model.fit(modelData.train.x, modelData.train.y,
                            batch_size=modelInstance.descriptor.batchSize,
                            epochs=modelInstance.descriptor.epochCount, callbacks=[earlyStop],
                            validation_data=(modelData.val.x, modelData.val.y)
                            )
        modelInstance.model.summary()
        score = modelInstance.model.evaluate(modelData.test.x, modelData.test.y, verbose=0)

        # score = [54.34, 98.55]
        modelInstance.score = score

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    except Exception as ex:
        print(ex)



    return modelInstance

# Batch size of 128 had about 90.4% accuracy.
# Thus, a batch size of 1000 was used where accuracy was about 91.5%. 
# Signifigantly higher batch sizes also decreased test accuracy.
# A batch size of 104,000 led to an accuracy of about 
# batchSize = 2000
# epochCount = 1000 #There is early stopping, so it won't reach 1000 epochs. This needs to be high.

inputShape = InputShape(28, 28)
numberOfClasses = 26

matFilepath = './emnist-letters.mat'

scoreFilepath = './models/results/scoreboard.csv'

modelDir = './models/'
archivePath = './models/archive/'
badPath = './models/archive/bad/'
threshholdAccuracy = 0.90

scoreBoard = ScoreBoard(scoreFilepath)
modelFiles = Utility.getFiles(modelDir)
for modelFile in modelFiles:
    descriptorPath = '%s%s'%(modelDir, modelFile)
    modelDescriptor = ModelDescriptor.create(descriptorPath, archivePath, badPath, inputShape, numberOfClasses, threshholdAccuracy)

    if (scoreBoard.isDone(modelDescriptor)):
        modelDescriptor.archive()
    else:
        modelInstance = ModelInstance.create(modelDescriptor)

        modelInstance = processNN(matFilepath, modelInstance)

        if (modelInstance.score == None):
            modelDescriptor.bad()
        else:
            scoreBoard.record(modelDescriptor, modelInstance.score)



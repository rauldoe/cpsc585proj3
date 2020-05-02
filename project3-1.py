from __future__ import print_function

# %tensorflow_version 2.x
# Comment it out if not connecting to Google Drive
# Please run pip install google.colab
# from google.colab import drive
import csv
import timeit

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# Importing the EMNIST letters
from scipy import io as sio

class LayerDescriptor:
    def __init__(self, inputShape, numberOfClasses, inputList):
        self.InputShape = inputShape
        self.NumberOfClasses = numberOfClasses
        self.InputList = inputList
        self.LayerType = inputList[0]

    def build(self):
        layerType = self.LayerType
        inputList = self.InputList
        inputCount = len(inputList)

        if (layerType == 'conv2d'):
            self.Activation = inputList[1]
            self.FilterCount = int(inputList[2])
            self.KernelSize = (int(inputList[3]), int(inputList[4]))

            if (inputCount > 5):
                layer = Conv2D(self.FilterCount, kernel_size=self.KernelSize, activation=self.Activation, input_shape=self.InputShape)
            else:
                layer = Conv2D(self.FilterCount, kernel_size=self.KernelSize, activation=self.Activation)

        elif (self.LayerType == 'maxpooling2d'):
            layer = MaxPooling2D(pool_size=(int(inputList[1]), int(inputList[2])))

        elif (self.LayerType == 'dropout'):
            layer = Dropout(float(inputList[1]))

        elif (self.LayerType == 'flatten'):
            layer = Flatten()

        elif (self.LayerType == 'dense'):
            self.Activation = inputList[1]

            if (inputCount > 2):
                layer = Dense(int(inputList[2]), activation=self.Activation)
            else:
                layer = Dense(self.NumberOfClasses, activation=self.Activation)
        
        return layer

    @staticmethod
    def create(inputShape, numberOfClasses, inputList):
        return LayerDescriptor(inputShape, numberOfClasses, inputList)

class Data2d:
    def __init__(self, xData, yData):
        self.x = xData
        self.y = yData

class InputShape:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.modified = None

    def t(self):
        return (self.rows, self.cols)

class ModelData:
    def __init__(self, train, test, val, inputShape):
        self.train = train
        self.test = test
        self.val = val
        self.inputShape = inputShape


def loadCsv(csvPath):
    descriptions = []
    with open(csvPath, newline='\r\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            descriptions.append(row)

    return descriptions

def loadModelData(matFilepath):
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

def buildLayers(descriptions, inputShape, numberOfClasses):
    descriptors = []
    layers = []

    for desc in descriptions:
        descriptor = LayerDescriptor.create(inputShape, numberOfClasses, desc)
        descriptors.append(descriptor)
        layers.append(descriptor.build())

    return layers


def buildModel(layers):
    model = Sequential()

    for layer in layers:
        model.add(layer)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
    
    return model


def processNN(matFilepath, inputShape, numberOfClasses, batchSize, epochCount, descriptorFilepath):
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    earlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
        )

    descriptions = loadCsv(descriptorFilepath)

    modelData = loadModelData(matFilepath)

    layers = buildLayers(descriptions, inputShape.modified, numberOfClasses)
    model = buildModel(layers)


    model.fit(modelData.train.x, modelData.train.y,
                        batch_size=batchSize,
                        epochs=epochCount, callbacks=[earlyStop],
                        validation_data=(modelData.val.x, modelData.val.y)
                        )
    score = model.evaluate(modelData.test.x, modelData.test.y, verbose=0)


    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score

# Batch size of 128 had about 90.4% accuracy.
# Thus, a batch size of 1000 was used where accuracy was about 91.5%. 
# Signifigantly higher batch sizes also decreased test accuracy.
# A batch size of 104,000 led to an accuracy of about 
batchSize = 2000
epochCount = 1000 #There is early stopping, so it won't reach 1000 epochs. This needs to be high.

inputShape = InputShape(28, 28)
numberOfClasses = 26

acivationsFilepath = './activations.csv'
acivations = loadCsv(acivationsFilepath)

matFilepath = './emnist-letters.mat'

# descriptorFilepath = './model.csv'
# score = processNN(matFilepath, inputShape, numberOfClasses, batchSize, epochCount, descriptorFilepath)

descriptorFilepath = './model1.csv'
score = processNN(matFilepath, inputShape, numberOfClasses, batchSize, epochCount, descriptorFilepath)

# escriptorFilepath = './model2.csv'
# score = processNN(matFilepath, inputShape, numberOfClasses, batchSize, epochCount, descriptorFilepath)


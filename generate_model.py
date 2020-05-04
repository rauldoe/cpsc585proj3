from __future__ import print_function

# %tensorflow_version 2.x

import csv
import timeit
import copy
import shutil
import random
import os.path
from os import path

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

class Generator:
    def __init__(self, modelDir, choiceDir):
        self.modelDir = modelDir
        self.choiceDir = choiceDir

        self.activations = None
        self.kernelRegularizers = None
        self.kernalSizes = None     
        self.layers = None
        self.losses = None
        self.optimizers = None
        self.filterCount = None
        self.maxPooling2d = None
        self.ranges = None

    def load(self):
        self.activations = Utility.readCsv(self.choiceDir+'activations.csv')
        self.kernelRegularizers = Utility.readCsv(self.choiceDir+'kernel_regularizers.csv')
        self.kernelRegularizers.append([''])
        self.kernalSizes = Utility.readCsv(self.choiceDir+'kernel_sizes.csv')       
        self.layers = Utility.readCsv(self.choiceDir+'layers.csv')
        self.losses = Utility.readCsv(self.choiceDir+'losses.csv')
        self.optimizers = Utility.readCsv(self.choiceDir+'optimizers.csv')
        self.filterCount = Utility.readCsv(self.choiceDir+'filter_count.csv')
        self.maxPooling2d = Utility.readCsv(self.choiceDir+'maxpooling2ds.csv')
        self.ranges = Utility.readCsv(self.choiceDir+'ranges.csv')

    def randomLayer(self, i, layerCount):

        if (i<layerCount-1):
            layerName = Utility.getRandom(self.layers)[0]
        else:
            allowed = ['conv2d', 'dense']
            layerName = Utility.getRandom(allowed)

        layer = LayerDescriptor(layerName)
        output = ''

        if (layer.layerName == 'conv2d'):
            layer.activationFunction = Utility.getRandom(self.activations)[0]
            layer.filterCount = int(Utility.getRandom(self.filterCount)[0])
            kernelSize = Utility.getRandom(self.kernalSizes)
            layer.kernelSize = (int(kernelSize[0]), int(kernelSize[1]))
            layer.kernelRegularizer = Utility.getRandom(self.kernelRegularizers)[0]
            if (layer.kernelRegularizer == ''):
                layer.kernelRegularizerValue = ''
            else:
                layer.kernelRegularizerValue = Utility.randomFloat(self.ranges, 'kernelRegularizerValue')
            output = '%s,%s,%d,%d,%d,%s,%s'%(layer.layerName, layer.activationFunction, layer.filterCount, layer.kernelSize[0], layer.kernelSize[1], layer.kernelRegularizer, layer.kernelRegularizerValue )
        elif (layer.layerName == 'maxpooling2d'):
            layer.maxPooling2d = Utility.getRandom(self.maxPooling2d)
            output = '%s,%d,%d'%(layer.layerName, int(layer.maxPooling2d[0]), int(layer.maxPooling2d[1]))
        elif (layer.layerName == 'dropout'):
            layer.dropout = Utility.randomFloat(self.ranges, 'dropout')
            output = '%s,%f'%(layer.layerName, layer.dropout)
        elif (layer.layerName == 'flatten'):
            output = '%s'%(layer.layerName)
        elif (layer.layerName == 'dense'):
            layer.activationFunction = Utility.getRandom(self.activations)[0]
            layer.filterCount = int(Utility.getRandom(self.filterCount)[0])
            output = '%s,%s,%d'%(layer.layerName, layer.activationFunction, layer.filterCount)
        else:
            layer.layerName = 'conv2d'           
            layer.activationFunction = Utility.getRandom(self.activations)[0]
            layer.filterCount = int(Utility.getRandom(self.filterCount)[0])
            layer.kernelRegularizer = Utility.getRandom(self.kernelRegularizers)[0]
            layer.kernelRegularizerValue = Utility.randomFloat(self.ranges, 'kernelRegularizerValue')
            output = '%s,%s,%d,%s,%f'%(layer.layerName, layer.activationFunction, layer.filterCount, layer.kernelRegularizer, layer.kernelRegularizerValue )
        
        return [layer, output]

    def random(self):
        md = ModelDescriptor()

        md.batchSize = Utility.randomInt(self.ranges, 'batchSize')
        md.loss = Utility.getRandom(self.losses)[0]
        md.optimizer = Utility.getRandom(self.optimizers)[0]
        md.layerCount = Utility.randomInt(self.ranges, 'layerCount')

        output = '%d,%d,%d,%s,%s'%(md.processCount, md.epochCount, md.batchSize, md.loss, md.optimizer)
        md.outputList.append(output)

        for i in range(md.layerCount):
            rl = self.randomLayer(i, md.layerCount)
            md.layers.append(rl[0])
            md.outputList.append(rl[1])

        return md

class LayerDescriptor:
    def __init__(self, layerName):
        self.layerName = layerName
        self.activationFunction = None
        self.filterCount = None
        self.kernelSize = None
        self.kernelRegularizer = None
        self.kernelRegularizerValue = None
        self.maxPooling2d = None
        self.dropout = None

class ModelDescriptor:
    def __init__(self):
        self.processCount = 3
        self.epochCount = 1000
        self.batchSize = None
        self.loss = None
        self.optimizer = None
        self.layerCount = None
        self.layers = []

        self.outputList = []

modelDir = './models/'
choiceDir = './models/choices/'
modelCount = 100

generator = Generator(modelDir, choiceDir)
generator.load()

for i in range(modelCount):
    md = generator.random()
    
    while True:
        modelFilepath = '%smodel-%s.csv'%(modelDir, Utility.generateRandom())
        if (not path.exists(modelFilepath)):
            Utility.writeList(modelFilepath, md.outputList)
            break

    print(str(i)+' generated')

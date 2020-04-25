from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Importing the EMNIST letters
from scipy import io as sio


# Batch size of 128 had about 90.4% accuracy.
# Thus, a batch size of 1000 was used where accuracy was about 91.5%. 
# Signifigantly higher batch sizes also decreased test accuracy.
# A batch size of 104,000 led to an accuracy of about 
batch_size = 1000
# num_classes = 10
num_classes = 26
epochs = 1000 #There is early stopping, so it won't reach 1000 epochs. This needs to be high.

# https://stackoverflow.com/questions/51125969/loading-emnist-letters-dataset/53547262#53547262
mat = sio.loadmat('emnist-letters.mat')
data = mat['dataset']

x_train = data['train'][0,0]['images'][0,0]
y_train = data['train'][0,0]['labels'][0,0]
x_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]

val_start = x_train.shape[0] - x_test.shape[0]
x_val = x_train[val_start:x_train.shape[0],:]
y_val = y_train[val_start:x_train.shape[0]]
x_train = x_train[0:val_start,:]
y_train = y_train[0:val_start]


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes, dtype='float32')

y_val = tf.keras.utils.to_categorical(y_val - 1, num_classes, dtype='float32')

model = Sequential()
# Sigmoid seemed to work better for test accuracy compared to relu. (sigmoid was getting 91% test accuracy compared to 89% for relu.)
# Sigmoid was slighly better than tanh, but both were about the same test accuracy (within a few tenths of a percent)
model.add(Dense(512, activation='sigmoid', input_shape=(784,)))
# Tried different dropout rates, but 0.2 seemed to work well and provided a modest improvement.
# (~0.5% test accuracy improvement compared to not using dropout at all)
model.add(Dropout(0.2))
# Compared to other numbers of neurons, this number seemed to work well (2000 hidden neurons)
model.add(Dense(2000, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs, callbacks=[earlyStop],
                    validation_data=(x_val, y_val)
                    )
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
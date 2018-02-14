'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import pandas as pd
from IPython import embed

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

store = pd.HDFStore('/home/karel/working/QLKNN-develop/unstable_training_gen2_7D_nions0_flat_filter7.h5') # Change this path!
target_df = store['efiITG_GB'].to_frame() # This one is relatively easy to train
input_df = store['input']
# Split the set somewhere. Optionally shuffle it first
ind = int(0.9 * len(target_df))
y_train = target_df.iloc[:ind, :]
y_test  = target_df.iloc[ind:, :]

# Use the index (row numbers) to select the correct rows from the input set
x_train = input_df.loc[y_train.index]
x_test  = input_df.loc[y_test.index]

# Maybe it's more convenient for you to work with numpy arrays.
# To extract them from pandas, just do 'x_train.values'

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

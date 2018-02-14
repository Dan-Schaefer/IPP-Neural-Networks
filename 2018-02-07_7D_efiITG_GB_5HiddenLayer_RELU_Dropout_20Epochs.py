#! /usr/bin/env python

'''
Trains 7D QuaLiKiz-NN with a single output (efiTG)
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam
import pandas
import numpy

# Define neural network parameters
batch_size = 128
num_classes = 1
epochs = 20

# Load Data (which is in HDF5 or .h5 format)
store = pandas.HDFStore("../../unstable_training_gen2_7D_nions0_flat_filter7.h5")
target_df = store['efiITG_GB'].to_frame()  # This one is relatively easy to train
input_df = store['input']


# Puts inputs and outputs in the same pandas dataframe. Also only keeps overlapping entries.
joined_dataFrame = target_df.join(input_df)

# Shuffles dataset
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(joined_dataFrame.index))

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df = shuffled_joined_dataFrame['efiITG_GB']

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df = shuffled_joined_dataFrame.drop('efiITG_GB', axis=1)

# Creates training dataset (90% of total data) for outputs
y_train = shuffled_clean_output_df.iloc[:int(
    numpy.round(len(shuffled_clean_output_df)*0.9))]

# Creates training dataset (90% of total data) for inputs
x_train = shuffled_clean_input_df.iloc[:int(
    numpy.round(len(shuffled_clean_input_df)*0.9))]

# Creates testing dataset (10% of total data) for outputs
y_test = shuffled_clean_output_df.iloc[int(
    numpy.round(len(shuffled_clean_output_df)*0.9)):]

# Creates testing dataset (1% of total data) for inputs
x_test = shuffled_clean_input_df.iloc[int(
    numpy.round(len(shuffled_clean_input_df)*0.9)):]

# Deletes pandas dataframes that are no longer needed
del target_df, input_df

# Closes the HDFStore. This is good practice.
store.close()

# Define neural network
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(7,)))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

# Add TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='TensorBoard_logs/', write_graph=True)

model.compile(loss='mean_squared_error',   # TODO
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# creates a HDF5 file 'my_model.h5'
model.save("2018-01-27_7D_efiITG_GB_5HiddenLayer_RELU_Dropout_20Epochs.h5")

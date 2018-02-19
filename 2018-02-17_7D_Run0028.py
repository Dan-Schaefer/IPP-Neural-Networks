#! /usr/bin/env python

'''
Trains 7D QuaLiKiz-NN with a single output (efiTG)
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, adam, Adam
from keras.initializers import TruncatedNormal
from keras import regularizers
import pandas
import numpy
import sys
import os

# Gets the current file name. Useful for procedurally generating output/log files.
file_name =  os.path.basename(sys.argv[0][:-3])

# Define neural network parameters
batch_size = 10
#num_classes = 1
epochs = 100

# Load Data (which is in HDF5 or .h5 format)
store = pandas.HDFStore("unstable_training_gen2_7D_nions0_flat_filter7.h5")
target_df = store['efiITG_GB'].to_frame()  # This one is relatively easy to train
input_df = store['input']

# Puts inputs and outputs in the same pandas dataframe.
# Also only keeps overlapping entries.
joined_dataFrame = target_df.join(input_df)

# Remove all negative values
joined_dataFrame = joined_dataFrame[(joined_dataFrame['efiITG_GB']>0)
        & (joined_dataFrame['Ati']>0)
        & (joined_dataFrame['Ate']>0)
        & (joined_dataFrame['An']>0)
        & (joined_dataFrame['qx']>0)
        & (joined_dataFrame['smag']>0)
        & (joined_dataFrame['x']>0)
        & (joined_dataFrame['Ti_Te']>0)]

# Shuffles dataset
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(
                                                joined_dataFrame.index))

# Normalizes data (no standardization necessary due to dataset design)
shuffled_joined_dataFrame = shuffled_joined_dataFrame/shuffled_joined_dataFrame \
        .max().astype(numpy.float64)

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
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.000001),
        input_shape=(7,)))
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.000001)))
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.000001)))
model.add(Dense(1,
        activation='linear'))
#model.add(keras.layers.normalization.BatchNormalization())
model.summary()

# Add CallBacks (including TensorBoard)
tbCallBack = keras.callbacks.TensorBoard(
        log_dir='TensorBoard_logs/' + str(file_name), write_graph=True)
EarlyStoppingCallBack = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.compile(loss='mean_squared_error',   #categorical_crossentropy
              #optimizer='adam',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    callbacks=[tbCallBack, EarlyStoppingCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Create output file
OutputFile = open("./Loss-Values/" +str(file_name) +".txt", "w+")
OutputFile.write("Test loss: " + str(score[0]) + "\n")
OutputFile.write("Test accuracy: " + str(score[1]))
OutputFile.close()

# creates a HDF5 file 'my_model.h5'
model.save("./Saved-Networks/" + str(file_name) +".h5")

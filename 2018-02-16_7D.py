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

# Define neural network parameters
batch_size = 50
#num_classes = 1
epochs = 20

# Load Data (which is in HDF5 or .h5 format)
store = pandas.HDFStore("unstable_training_gen2_7D_nions0_flat_filter7.h5")
target_df = store['efiITG_GB'].to_frame()  # This one is relatively easy to train
input_df = store['input']


# Puts inputs and outputs in the same pandas dataframe. Also only keeps overlapping entries.
joined_dataFrame = target_df.join(input_df)
print("print joined_dataFrame - original")
print(joined_dataFrame)

# Remove all negative values
joined_dataFrame = joined_dataFrame[(joined_dataFrame['efiITG_GB']>0)
        & (joined_dataFrame['Ati']>0)
        & (joined_dataFrame['Ate']>0)
        & (joined_dataFrame['An']>0)
        & (joined_dataFrame['qx']>0)
        & (joined_dataFrame['smag']>0)
        & (joined_dataFrame['x']>0)
        & (joined_dataFrame['Ti_Te']>0)]
print("print joined_dataFrame - negatives removed")
print(joined_dataFrame)


# Shuffles dataset
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(joined_dataFrame.index))
print("print shuffled_joined_dataFrame - pre-normalization")
print(shuffled_joined_dataFrame)

# Normalizes data (no standardization necessary due to dataset design)
shuffled_joined_dataFrame = shuffled_joined_dataFrame/shuffled_joined_dataFrame.max().astype(numpy.float64)
print("print shuffled_joined_dataFrame - post-normalization")
print(shuffled_joined_dataFrame)

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df = shuffled_joined_dataFrame['efiITG_GB']
print("print shuffled_clean_output_df")
print(shuffled_clean_output_df)

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df = shuffled_joined_dataFrame.drop('efiITG_GB', axis=1)
print("print shuffled_clean_input_df")
print(shuffled_clean_input_df)

# Creates training dataset (90% of total data) for outputs
y_train = shuffled_clean_output_df.iloc[:int(
    numpy.round(len(shuffled_clean_output_df)*0.9))]
print("print y_train")
print(y_train)

# Creates training dataset (90% of total data) for inputs
x_train = shuffled_clean_input_df.iloc[:int(
    numpy.round(len(shuffled_clean_input_df)*0.9))]
print("print x_train")
print(x_train)

# Creates testing dataset (10% of total data) for outputs
y_test = shuffled_clean_output_df.iloc[int(
    numpy.round(len(shuffled_clean_output_df)*0.9)):]
print("print y_test")
print(y_test)

# Creates testing dataset (1% of total data) for inputs
x_test = shuffled_clean_input_df.iloc[int(
    numpy.round(len(shuffled_clean_input_df)*0.9)):]
print("print x_test")
print(x_test)

# Deletes pandas dataframes that are no longer needed
del target_df, input_df

# Closes the HDFStore. This is good practice.
store.close()

# Define neural network
model = Sequential()
model.add(Dense(30, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=regularizers.l2(0.1), input_shape=(7,)))
model.add(Dense(30, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(30, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1, activation='linear'))
#model.add(keras.layers.normalization.BatchNormalization())
model.summary()

# Add TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='TensorBoard_logs/2018-02-16_7D/', write_graph=True)
#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

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
                    callbacks=[tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# creates a HDF5 file 'my_model.h5'
model.save("2018-02-16_7D.h5")

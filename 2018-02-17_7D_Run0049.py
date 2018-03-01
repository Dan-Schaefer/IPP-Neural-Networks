#! /usr/bin/env python

'''
Trains 7D QuaLiKiz-NN with all output fluxes
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, adam, Adam
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras import backend as K
import pandas
import numpy
import sys
import os

# Define new Metric: rmse = Root Mean Square Error
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square( y_true-y_pred )))

# Gets the current file name. Useful for procedurally generating output/log files.
file_name =  os.path.basename(sys.argv[0][:-3])

# Define neural network parameters
batch_size = 10
#num_classes = 1
epochs = 100

# Load Data (which is in HDF5 or .h5 format)
store = pandas.HDFStore("unstable_training_gen2_7D_nions0_flat_filter7.h5")
target_df_1 = store['pfiITG_GB'].to_frame()
target_df_2 = store['pfiTEM_GB'].to_frame()
target_df_3 = store['pfeITG_GB'].to_frame()
target_df_4 = store['pfeTEM_GB'].to_frame()
target_df_5 = store['efiITG_GB'].to_frame()
target_df_6 = store['efiTEM_GB'].to_frame()
target_df_7 = store['efeETG_GB'].to_frame()
target_df_8 = store['efeITG_GB'].to_frame()
target_df_9 = store['efeTEM_GB'].to_frame()
input_df = store['input']

# Puts inputs and outputs in the same pandas dataframe.
# Also only keeps overlapping entries.
joined_dataFrame = target_df_1.join(input_df)
joined_dataFrame = target_df_2.join(joined_dataFrame)
joined_dataFrame = target_df_3.join(joined_dataFrame)
joined_dataFrame = target_df_4.join(joined_dataFrame)
joined_dataFrame = target_df_5.join(joined_dataFrame)
joined_dataFrame = target_df_6.join(joined_dataFrame)
joined_dataFrame = target_df_7.join(joined_dataFrame)
joined_dataFrame = target_df_8.join(joined_dataFrame)
joined_dataFrame = target_df_9.join(joined_dataFrame)

print("joined_dataFrame")
print(joined_dataFrame)
print(joined_dataFrame.shape)


# Remove all negative values
joined_dataFrame = joined_dataFrame[(joined_dataFrame['pfiITG_GB']>0) #
        & (joined_dataFrame['pfiTEM_GB']>0) #
        & (joined_dataFrame['pfeITG_GB']>0) #
        & (joined_dataFrame['pfeTEM_GB']>0) #
        & (joined_dataFrame['efiITG_GB']>0) #
        & (joined_dataFrame['efiTEM_GB']>0) #
        & (joined_dataFrame['efeETG_GB']>0) #
        & (joined_dataFrame['efeITG_GB']>0) #
        & (joined_dataFrame['efeTEM_GB']>0) #
        & (joined_dataFrame['Ati']>0)
        & (joined_dataFrame['Ate']>0)
        & (joined_dataFrame['An']>0)
        & (joined_dataFrame['qx']>0)
        & (joined_dataFrame['smag']>0)
        & (joined_dataFrame['x']>0)
        & (joined_dataFrame['Ti_Te']>0)]
print("joined_dataFrame")
print(joined_dataFrame)
print(joined_dataFrame.shape)

# Shuffles dataset
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(
                                                joined_dataFrame.index))

# Normalizes data (no standardization necessary due to dataset design)
shuffled_joined_dataFrame = shuffled_joined_dataFrame/shuffled_joined_dataFrame \
        .max().astype(numpy.float64)
print("shuffled_joined_dataFrame")
print(shuffled_joined_dataFrame)

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df = shuffled_joined_dataFrame.copy()
shuffled_clean_output_df.drop(['Ati', 'Ate', 'An', 'qx', 'smag', 'x', 'Ti_Te'], axis=1, inplace=True)
print("shuffled_clean_output_df")
print(shuffled_clean_output_df)

print("shuffled_joined_dataFrame")
print(shuffled_joined_dataFrame)

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df = shuffled_joined_dataFrame.copy()
shuffled_clean_input_df.drop(['pfiITG_GB', 'pfiTEM_GB', 'pfeITG_GB', 'pfeTEM_GB', 'efiITG_GB', 'efiTEM_GB', 'efeETG_GB', 'efeITG_GB', 'efeTEM_GB'], axis=1, inplace=True)
print("shuffled_clean_input_df")
print(shuffled_clean_input_df)

# Creates training dataset (90% of total data) for outputs
y_train = shuffled_clean_output_df.iloc[:int(
    numpy.round(len(shuffled_clean_output_df)*0.9))]

# Creates training dataset (90% of total data) for inputs
x_train = shuffled_clean_input_df.iloc[:int(
    numpy.round(len(shuffled_clean_input_df)*0.9))]

# Creates testing dataset (10% of total data) for outputs
y_test = shuffled_clean_output_df.iloc[int(
    numpy.round(len(shuffled_clean_output_df)*0.9)):]

# Creates testing dataset (10% of total data) for inputs
x_test = shuffled_clean_input_df.iloc[int(
    numpy.round(len(shuffled_clean_input_df)*0.9)):]

# Deletes pandas dataframes that are no longer needed
del target_df_1, \
    target_df_2, \
    target_df_3, \
    target_df_4, \
    target_df_5, \
    target_df_6, \
    target_df_7, \
    target_df_8, \
    target_df_9, \
    input_df

# Closes the HDFStore. This is good practice.
store.close()

# Define neural network
model = Sequential()
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.00000001),
        input_shape=(7,)))
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.00000001)))
model.add(Dense(30,
        activation='tanh',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        kernel_regularizer=regularizers.l2(0.00000001)))
model.add(Dense(9,
        activation='linear'))
#model.add(keras.layers.normalization.BatchNormalization())
model.summary()

# Add CallBacks (including TensorBoard)
tbCallBack = keras.callbacks.TensorBoard(
        log_dir='TensorBoard_logs/' + str(file_name), write_graph=True)
EarlyStoppingCallBack = keras.callbacks.EarlyStopping(
        monitor='val_rmse', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='mean_squared_error',   #categorical_crossentropy
              #optimizer='adam',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy', "mae", "mean_squared_error", rmse])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    callbacks=[tbCallBack]) #EarlyStoppingCallBack
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("score")
print(score)

print("model.metrics_names")
print(model.metrics_names)

# Create output file
OutputFile = open("./Loss-Values/" +str(file_name) +".txt", "w+")
OutputFile.write("Test loss: " + str(score[0]) + "\n")
OutputFile.write("Test accuracy: " + str(score[1]) + "\n")
OutputFile.write("val_mean_absolute_error: " +str(score[2]) + "\n")
OutputFile.write("val_mean_squared_error: " +str(score[3]) + "\n")
OutputFile.write("RMSE: " +str(score[4]) + "\n")
OutputFile.close()

# creates a HDF5 file 'my_model.h5'
model.save("./Saved-Networks/" + str(file_name) +".h5")

'''
Late Fusion Module (test) - Functional API
'''

# Multiple Inputs
import keras
from keras.optimizers import RMSprop, adam, Adam
from keras.initializers import TruncatedNormal, glorot_normal
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import regularizers
from keras import backend as K
import pandas
import numpy
import sys
import os
from copy import deepcopy

keras.backend.clear_session()





# Define new Metric: rmse = Root Mean Square Error
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square( y_true-y_pred )))

# Define custon LateFusionActivation function
def custom_activation(LateFusionActivation):
    return TODO

# Gets the current file name. Useful for procedurally generating output/log files.
file_name =  os.path.basename(sys.argv[0][:-3])

# Define neural network parameters
batch_size = 10
#num_classes = 1
epochs = 100

# Load Data (which is in HDF5 or .h5 format)
store = pandas.HDFStore("unstable_training_gen3_7D_nions0_flat_filter8.h5")
target_df = store['/output/efeETG_GB'].to_frame()  # This one is relatively easy to train
input_df = store['input']

# Puts inputs and outputs in the same pandas dataframe.
# Also only keeps overlapping entries.
joined_dataFrame = target_df.join(input_df)

# Make a copy of joined_dataFrame for later use
joined_dataFrame_original = deepcopy(joined_dataFrame)

# Make a copy of joined_dataFrame for branch1
joined_dataFrame_1 = deepcopy(joined_dataFrame)

# Make a copy of joined_dataFrame for branch2
joined_dataFrame_2 = deepcopy(joined_dataFrame)

# *************************************************************************** #
# Normalize data by standard deviation and mean-centering the data
# Standard configuration
joined_dataFrame['efeETG_GB'] = (joined_dataFrame['efeETG_GB'] - joined_dataFrame['efeETG_GB'].mean()) / joined_dataFrame['efeETG_GB'].std()
joined_dataFrame['Ati'] = (joined_dataFrame['Ati'] - joined_dataFrame['Ati'].mean()) / joined_dataFrame['Ati'].std()
joined_dataFrame['Ate'] = (joined_dataFrame['Ate'] - joined_dataFrame['Ate'].mean()) / joined_dataFrame['Ate'].std()
joined_dataFrame['An'] = (joined_dataFrame['An'] - joined_dataFrame['An'].mean()) / joined_dataFrame['An'].std()
joined_dataFrame['q'] = (joined_dataFrame['q'] - joined_dataFrame['q'].mean()) / joined_dataFrame['q'].std()
joined_dataFrame['smag'] = (joined_dataFrame['smag'] - joined_dataFrame['smag'].mean()) / joined_dataFrame['smag'].std()
joined_dataFrame['x'] = (joined_dataFrame['x'] - joined_dataFrame['x'].mean()) / joined_dataFrame['x'].std()
joined_dataFrame['Ti_Te'] = (joined_dataFrame['Ti_Te'] - joined_dataFrame['Ti_Te'].mean()) / joined_dataFrame['Ti_Te'].std()

# Shuffles dataset
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(
                                                joined_dataFrame.index))

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df = shuffled_joined_dataFrame['efeETG_GB']

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df = shuffled_joined_dataFrame.drop('efeETG_GB', axis=1)

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
# *************************************************************************** #

# *************************************************************************** #
# Normalize data by standard deviation and mean-centering the data
# Inputs for branch1
joined_dataFrame_1['efeETG_GB'] = (joined_dataFrame_1['efeETG_GB'] - joined_dataFrame_1['efeETG_GB'].mean()) / joined_dataFrame_1['efeETG_GB'].std()
joined_dataFrame_1['Ati'] = (joined_dataFrame_1['Ati'] - joined_dataFrame_1['Ati'].mean()) / joined_dataFrame_1['Ati'].std()
# joined_dataFrame_1['Ate'] = (joined_dataFrame_1['Ate'] - joined_dataFrame_1['Ate'].mean()) / joined_dataFrame_1['Ate'].std()
joined_dataFrame_1['An'] = (joined_dataFrame_1['An'] - joined_dataFrame_1['An'].mean()) / joined_dataFrame_1['An'].std()
joined_dataFrame_1['q'] = (joined_dataFrame_1['q'] - joined_dataFrame_1['q'].mean()) / joined_dataFrame_1['q'].std()
joined_dataFrame_1['smag'] = (joined_dataFrame_1['smag'] - joined_dataFrame_1['smag'].mean()) / joined_dataFrame_1['smag'].std()
joined_dataFrame_1['x'] = (joined_dataFrame_1['x'] - joined_dataFrame_1['x'].mean()) / joined_dataFrame_1['x'].std()
joined_dataFrame_1['Ti_Te'] = (joined_dataFrame_1['Ti_Te'] - joined_dataFrame_1['Ti_Te'].mean()) / joined_dataFrame_1['Ti_Te'].std()

print("joined_dataFrame_1.shape")
print(joined_dataFrame_1.shape)

# Shuffles dataset
shuffled_joined_dataFrame_1 = joined_dataFrame_1.reindex(numpy.random.permutation(
                                                joined_dataFrame_1.index))

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df_1 = shuffled_joined_dataFrame_1['efeETG_GB']

print("shuffled_clean_output_df_1.shape")
print(shuffled_clean_output_df_1.shape)

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df_1 = shuffled_joined_dataFrame_1.drop('efeETG_GB', axis=1)

print("shuffled_clean_input_df_1.shape")
print(shuffled_clean_input_df_1.shape)

# Creates training dataset (90% of total data) for outputs
y_train_1 = shuffled_clean_output_df_1.iloc[:int(
    numpy.round(len(shuffled_clean_output_df_1)*0.9))]

# Creates training dataset (90% of total data) for inputs
x_train_1 = shuffled_clean_input_df_1.iloc[:int(
    numpy.round(len(shuffled_clean_input_df_1)*0.9))]

# Creates testing dataset (10% of total data) for outputs
y_test_1 = shuffled_clean_output_df_1.iloc[int(
    numpy.round(len(shuffled_clean_output_df_1)*0.9)):]

# Creates testing dataset (10% of total data) for inputs
x_test_1 = shuffled_clean_input_df_1.iloc[int(
    numpy.round(len(shuffled_clean_input_df_1)*0.9)):]
# *************************************************************************** #

# *************************************************************************** #
# Normalize data by standard deviation and mean-centering the data
# Inputs for branch2
joined_dataFrame_2['Ate'] = (joined_dataFrame_2['Ate'] - joined_dataFrame_2['Ate'].mean()) / joined_dataFrame_2['Ate'].std()

# Shuffles dataset
shuffled_joined_dataFrame_2 = joined_dataFrame_2.reindex(numpy.random.permutation(
                                                joined_dataFrame_2.index))

# Creates a pandas dataframe for the outputs
shuffled_clean_output_df_2 = shuffled_joined_dataFrame_2['efeETG_GB']

# Creates a pandas dataframe for the inputs
shuffled_clean_input_df_2 = shuffled_joined_dataFrame_2.drop('efeETG_GB', axis=1)

# Creates training dataset (90% of total data) for outputs
y_train_2 = shuffled_clean_output_df_2.iloc[:int(
    numpy.round(len(shuffled_clean_output_df_2)*0.9))]

# Creates training dataset (90% of total data) for inputs
x_train_2 = shuffled_clean_input_df_2.iloc[:int(
    numpy.round(len(shuffled_clean_input_df_2)*0.9))]

# Creates testing dataset (10% of total data) for outputs
y_test_2 = shuffled_clean_output_df_2.iloc[int(
    numpy.round(len(shuffled_clean_output_df_2)*0.9)):]

# Creates testing dataset (10% of total data) for inputs
x_test_2 = shuffled_clean_input_df_2.iloc[int(
    numpy.round(len(shuffled_clean_input_df_2)*0.9)):]
# *************************************************************************** #

# Deletes pandas dataframes that are no longer needed
del target_df, input_df

# Closes the HDFStore. This is good practice.
store.close()





print(x_test_1.shape)



# branch1
visible_branch1 = Input(shape=(6,))
hidden1_branch1 = Dense(30)(visible_branch1)
hidden2_branch1 = Dense(30)(hidden1_branch1)

# branch2
visible_branch2 = Input(shape=(1,))

# merge input models
merge = concatenate([hidden2_branch1, visible_branch2])

# interpretation model
output = Dense(1, activation='sigmoid')(merge)

model = Model(inputs=[visible_branch1, visible_branch2], outputs=output)

# summarize layers
print(model.summary())

# plot graph
# plot_model(model, 'ModelPlots/' + str(file_name) + 'model_plot.png')




model.compile(loss='mean_squared_error',   #categorical_crossentropy
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=["mae", "mean_squared_error", rmse])

# Add CallBacks (including TensorBoard)
tbCallBack = keras.callbacks.TensorBoard(
        log_dir='TensorBoard_logs/' + str(file_name), write_graph = False, write_images=False, write_grads=False)
EarlyStoppingCallBack = keras.callbacks.EarlyStopping(
        monitor='val_rmse', min_delta=0, patience=15, verbose=0, mode='auto')

history = model.fit([x_train_1, x_train_2],
                    y = y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    # validation_data=([x_test_branch1, x_test_branch2], y_test),
                    callbacks=[tbCallBack, EarlyStoppingCallBack])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('val_mean_absolute_error:', score[1])

print("score")
print(score)

print("model.metrics_names")
print(model.metrics_names)

# creates a HDF5 file 'my_model.h5'
model.save("./Saved-Networks/" + str(file_name) +".h5")

# Create output file
OutputFile = open("./Loss-Values/" +str(file_name) +".txt", "w+")
OutputFile.write("Test loss: " + str(score[0]) + "\n")
OutputFile.write("val_mean_absolute_error: " +str(score[1]) + "\n")
OutputFile.write("val_mean_squared_error: " +str(score[2]) + "\n")
OutputFile.write("RMSE: " +str(score[3]) + "\n")
OutputFile.close()

del history
del model

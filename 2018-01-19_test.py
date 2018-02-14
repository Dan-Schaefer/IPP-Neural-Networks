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

import pandas
import numpy

# from IPython import embed

batch_size = 128
num_classes = 1
epochs = 20

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

store = pandas.HDFStore("C:/Users/danie/Documents/Masterarbeit-IPP/Python/Keras"
                        "/unstable_training_gen2_7D_nions0_flat_filter7.h5")
target_df = store['efiITG_GB'].to_frame()  # This one is relatively easy to train
input_df = store['input']

'''
print("Checkpoint 1")
print(store)
print("Checkpoint 2")
print(store['efiITG_GB'].shape)
print("Checkpoint 3 - store['efiITG_GB']")
print(store['efiITG_GB'])
print("Checkpoint 3.5")
print(target_df.head)
print("Checkpoint 4")
print(target_df)
print("Checkpoint 4.5")
print(input_df)
print("Checkpoint 5")
print(store['input'].shape)
'''

print("Checkpoint 5.1 - joined_dataFrame")
joined_dataFrame = target_df.join(input_df)
print(joined_dataFrame)
print(joined_dataFrame.shape)

print("Checkpoint 5.2 - print(shuffled_joined_dataFrame)")
shuffled_joined_dataFrame = joined_dataFrame.reindex(numpy.random.permutation(joined_dataFrame.index))
print(shuffled_joined_dataFrame)

#output_clean_df = [joined_dataFrame.loc[i][0] for i in joined_dataFrame.index]
#print(output_clean_df)
#inputs_clean_df =[joined_dataFrame[i][1:7] for i in joined_dataFrame.index ]
#print(output_clean_df)
#print(inputs_clean_df)

print("Checkpoint 5.3 - print(shuffled_clean_output_df)")
shuffled_clean_output_df = shuffled_joined_dataFrame['efiITG_GB']
print(shuffled_clean_output_df)

print("Checkpoint 5.4 - print(shuffled_clean_input_df)")
shuffled_clean_input_df = shuffled_joined_dataFrame.drop('efiITG_GB', axis=1)
print(shuffled_clean_input_df)


print("Checkpoint 5.5 - print(shuffled_clean_output_df.iloc[0])")
y_train = shuffled_clean_output_df.iloc[:int(
    numpy.round(len(shuffled_clean_output_df)*0.9))]
print(shuffled_clean_output_df.iloc[:int(
    numpy.round(len(shuffled_clean_output_df)*0.9))])

x_train = shuffled_clean_input_df.iloc[:int(
    numpy.round(len(shuffled_clean_input_df)*0.9))]

y_test = shuffled_clean_output_df.iloc[int(
    numpy.round(len(shuffled_clean_output_df)*0.9)):]
print(len(y_test) + len(y_train) == len(shuffled_clean_output_df))

x_test = shuffled_clean_input_df.iloc[int(
    numpy.round(len(shuffled_clean_input_df)*0.9)):]

print(len(x_test) + len(x_train) == len(shuffled_clean_input_df))

'''
print("Checkpoint 5.6 - print(shuffled_clean_output_df.loc[0])")
print(shuffled_clean_output_df.loc[0])
'''
'''
print("Checkpoint 5.7 - print(y_train)")
y_train = shuffled_clean_output_df.drop([shuffled_clean_output_df.iloc[0], shuffled_clean_output_df.iloc[0.1 * len(shuffled_clean_output_df)]])
print(y_train)
'''

del target_df, input_df

'''
# Split the set somewhere. Optionally shuffle it first
ind = int(0.9 * len(target_df))
y_train = target_df.iloc[:ind, :]
y_test = target_df.iloc[ind:, :]

# Use the index (row numbers) to select the correct rows from the input set
x_train = input_df.loc[y_train.index]
x_test = input_df.loc[y_test.index]
'''

print("Checkpoint 5.9")
print(y_train)
print("Checkpoint 6")
print(y_test)
print("Checkpoint 7")
print(x_train)
print("Checkpoint 8")
print(x_test)
print("Checkpoint 9")
store.close()
print("Checkpoint 10")

'''
# Maybe it's more convenient for you to work with numpy arrays.
# To extract them from pandas, just do 'x_train.values'

x_train = x_train.reshape(60000, 7)
x_test = x_test.reshape(10000, 7)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
'''
'''
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''

print("Checkpoint 11")

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(7,)))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# Add TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

model.compile(loss='mean_squared_error',   # TODO
              optimizer=RMSprop(),
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

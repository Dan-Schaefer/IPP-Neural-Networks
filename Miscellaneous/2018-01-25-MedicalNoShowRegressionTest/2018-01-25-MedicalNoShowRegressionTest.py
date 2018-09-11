'''
Taken from the website: https://medium.com/rocknnull/playing-with-machine-learning-a-practical-example-using-keras-tensorflow-790375cd1abb

The puropse of this code is to see if I can get a neural network that solves a
regression problem (as opposed to a classification problem) that gets a
non-zero accuracy.
'''

# Preprocess the dataset
import pandas as pds

dataframeX = pds.read_csv("C:\\Users\\danie\\Documents\\Masterarbeit-IPP\\Python\\Keras\\2018-01-25-MedicalNoShowRegressionTest\\No-show-Issue-Comma-300k.csv", usecols=[0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13])
dataframeY = pds.read_csv("C:\\Users\\danie\\Documents\\Masterarbeit-IPP\\Python\\Keras\\2018-01-25-MedicalNoShowRegressionTest\\No-show-Issue-Comma-300k.csv", usecols=[5])

print(dataframeX.head())
print(dataframeY.head())

def weekdayToInt(weekday):
    return {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }[weekday]

def genderToInt(gender):
    if gender == 'M':
        return 0
    else:
        return 1

def statusToInt(status):
    if status == 'Show-Up':
        return 1
    else:
        return 0

dataframeX.DayOfTheWeek = dataframeX.DayOfTheWeek.apply(weekdayToInt)
dataframeX.Gender = dataframeX.Gender.apply(genderToInt)
dataframeY.Status = dataframeY.Status.apply(statusToInt)

print(dataframeX.head())
print(dataframeY.head())

# The Neural Network
# 1
import numpy as np
seed = 7
np.random.seed(seed)

# 2
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_shape=(11,), init='uniform', activation='sigmoid'))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.summary()

# 3
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# 4
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(dataframeX.values, dataframeY.values, epochs=9, batch_size=50,  verbose=1, validation_split=0.3, callbacks=[tbCallBack])

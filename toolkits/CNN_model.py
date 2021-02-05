#########################################
# This script shows the CNN model used in SteroidXtract.
# Shipei Xing, Oct 26, 2020
# Copyright @ The University of British Columbia
#########################################
# import libraries
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# CNN model
def CNN_model():
    classifier = Sequential()
    classifier.add(Conv1D(32, 3, activation = 'relu', input_shape = (1500,1)))
    classifier.add(Conv1D(32, 3, activation = 'relu'))
    classifier.add(MaxPooling1D(pool_size = 2, strides = 2))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 400, activation = 'relu'))
    classifier.add(Dense(units = 400, activation = 'relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return classifier
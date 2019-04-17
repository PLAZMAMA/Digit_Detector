from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.constraints import maxnorm
import numpy as np

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

def flat(arr): #flattens the data arr
    l = []
    for array in arr:
        l.append(array.flatten())
    return(np.array(l))

x_train = flat(x_train)
x_test = flat(x_test)

x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

###creating and training the nueral net
model = Sequential()
model.add(Dense(392, activation = "relu", input_dim = 784))
model.add(Dropout(0.5))
model.add(Dense(196, activation = "relu", kernel_constraint = maxnorm(3)))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size = 128, epochs = 3)


print(model.evaluate(x_test, y_test, batch_size = 128)) #testing to see if the model acually works and not overfitting

model.save("num_class_model.h5")
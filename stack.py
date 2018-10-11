import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
with open('datasets\input.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    X= [row for row in reader]
data = [[float(x)/100 for x in row] for row in X]
for row in data:
    row[5] /=10000000
    row[0] -= 20
    row[1] -= 20
    row[2] -= 20
    row[3] -= 20
    row[4] -= 20
X = np.array(data) * 4
print(X[0])
print(X.shape)
with open('datasets\output.csv','r', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    Y= [row for row in reader]
data = [[float(x)/100 -20 for x in row] for row in Y]
Y = np.array(data) * 4
print(Y[0])
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
model = Sequential()
model.add(Dense(16, activation='tanh', input_shape = (6,), kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(32, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(32, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(16, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(8, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(1, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.summary()
model.compile(loss = keras.losses.mean_squared_error,
            optimizer = keras.optimizers.Adadelta(),
            metrics = ["accuracy"])
hist = model.fit(X_train, y_train,
            batch_size = 128,
            epochs = 500,
            verbose = 1,
            validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
y_test_predict = model.predict(X_test[0:10])
print(X_test[0:10])
print(y_test[0:10])
print(y_test_predict)
print(y_test_predict.shape)
y_true_test = (y_test[0:10] / 4 + 20) * 100
print(y_true_test)
y_true_test_predict = (y_test_predict / 4 + 20) * 100
print(y_true_test_predict)  
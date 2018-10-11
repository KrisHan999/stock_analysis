# 导入包
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置归一化参数epsilon
epsi = 10e-6

# 读取input文件
with open('datasets\input.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    X= [row for row in reader]
# 将str类型转为float    
data = [[float(x) for x in row] for row in X]
# 转为numpy.array类型
X = np.array(data)
# 得到平均值和方差用来归一化输入
mean_X = np.mean(X, axis = 0)
var_X = np.var(X, axis = 0)
print("Mean_X:")
print(mean_X)
print("Variance_X:")
print(var_X)
print('\n')
# 对X归一化
X -= mean_X
X /= np.sqrt(var_X + epsi)
print(X[0])
print(X.shape)

# 读取target文件
with open('datasets\output.csv','r', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    Y= [row for row in reader]
# 将str类型转为float  
data = [[float(x) for x in row] for row in Y]
# 转为numpy.array类型
Y = np.array(data)
# 求得平均值和方差来进行归一化
mean_Y = np.mean(Y, axis = 0)
var_Y = np.var(Y, axis=0)
print("Mean_Y:")
print(mean_Y)
print("Variance_Y:")
print(var_Y)
print('\n')
# 对Y进行归一化
Y -= mean_Y
Y /= np.sqrt(var_Y + epsi)
print(Y[0])
print(Y.shape)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# 构建神经网络
model = Sequential()
model.add(Dense(16, activation='tanh', input_shape = (6,), kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(32, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(32, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(16, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(8, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.summary()

# 编译网络并进行训练
model.compile(loss = keras.losses.mean_squared_error,
            optimizer = keras.optimizers.Adadelta(),
            metrics = ["accuracy"])
hist = model.fit(X_train, y_train,
            batch_size = 128,
            epochs = 500,
            verbose = 1,
            validation_data = (X_test, y_test))
# 测试网络
score = model.evaluate(X_test, y_test)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# 进行预测
y_test_predict = model.predict(X[::2])

# Y真实值
y_true_test = Y[::2] * np.sqrt(var_Y + epsi) + mean_Y 

# Y预测值
y_true_test_predict = y_test_predict * np.sqrt(var_Y + epsi) + mean_Y

# 画图比较真实值和预测值
x_coor_len = len(y_true_test_predict)
x_coor_len
x_coor = range(x_coor_len)
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.plot(x_coor, y_true_test_predict, 'ro-', label='predict_value')
plt.plot(x_coor, y_true_test, 'bo-', label = 'real_value')
plt.legend()

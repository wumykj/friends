import re
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Activation,Dense,Flatten,Conv2D,Conv1D,MaxPooling2D,MaxPooling1D
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
columns = ['vehicle_speed','throttle_percentage','acc_calculated']
data = pd.read_csv('../data/brake_only.csv',names=columns)
def obtain_dataset(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        # next(f)
        lines = f.readlines()
        X, y ,data= [], [] ,[]
        seq = re.compile(",")
        for l in lines:
            l1 = seq.split(l.strip())
            l = [float(x) for x in l1]
            x, Y = l[1:len(l)-1], l[-1]
            X.append(x)
            y.append(Y)
            data.append(l[1:])
    print(X[0:10],y[0:10],data[0:10])
    # print(len(X),len(y),X[0:3],y[0:3])#4429 4429 [[23.2093, 0.25], [23.1467, 0.25], [23.09, 0.25]]
    # # [-0.121111111, -0.331388889, -0.503333333]
    return np.array(X), np.array(y), np.array(data)

def obtain_dataset1(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        # next(f)
        lines = f.readlines()
        X, y ,data= [], [] ,[]
        seq = re.compile(",")
        for l in lines:
            l1 = seq.split(l.strip())
            l = [float(x) for x in l1]
            x, Y = l[1:len(l)-1], l[-1]
            X.append(x)
            y.append(Y)
            data.append(l[1:])
    print(X[0:10],y[0:10],data[0:10])
    # print(len(X),len(y),X[0:3],y[0:3])#4429 4429 [[23.2093, 0.25], [23.1467, 0.25], [23.09, 0.25]]
    # # [-0.121111111, -0.331388889, -0.503333333]
    return X, y

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    :param data: Sequence of observations as a list or NumPy array.
    :param n_in: Number of lag observations as input (X).
    :param n_out: Number of observations as output (y).
    :param dropnan: Boolean whether or not to drop rows with NaN values.
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print(agg)
    # agg.drop(agg.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    print(len(agg))
    return agg

def seg_dataset(X,y,test_size):
    # set_aside
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def seg_dataset1(agg,train_size=0.6,valid_size=0.2):
    """
    :param agg: 输入的表格形式
    :param train_size: 选取前60%的数据作为训练集
    :param valid_size: 中间20%作为验证集
    :return:
    """
    n = len(agg)
    train_number = int(n*train_size)
    valid_number = int(n*valid_size)
    values = agg.values
    train = values[:train_number, :]
    valid = values[train_number:train_number + valid_number, :]
    test = values[train_number + valid_number:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    return train_X, valid_X, test_X, train_y, valid_y, test_y

def model_acc(y_pre,y_test,error):
    count = 0
    for i ,j in zip(y_pre,y_test):
        k = max(abs(i-j)/i,abs(i-j)/j)
        if k < error: count += 1
    return count/len(y_pre)


def model_run(name,file_path,error):
    # 其他模型如MLP,logic回归等
    # data pre
    X, y = obtain_dataset1(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # X_train, X_test, y_train, y_test = X,X,y,y
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    # select model calculate acc
    clf = Model(name)
    print(X_train[0:10], y_train[0:10])
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    print(y_pre[0:10])
    print(y_test[0:10])

    acc = model_acc(y_pre,y_test,error)
    print(acc)
    return acc

def model_run1(file_path,error):
    # LSTM模型
    # dataset_pre
    columns = ['vehicle_speed', 'throttle_percentage', 'acc_calculated']
    data = pd.read_csv(file_path, names=columns)
    agg = series_to_supervised(data, n_in=1, n_out=1, dropnan=True)
    values = agg.values
    train_X, valid_X, test_X, train_y, valid_y, test_y = seg_dataset1(agg, train_size=0.6, valid_size=0.2)
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # lstm_model_init
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    lstm = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(valid_X, valid_y), verbose=2,
                     shuffle=False)
    # plot history
    # plt.plot(lstm.history['loss'], label='train')
    # plt.plot(lstm.history['val_loss'], label='valid')
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=(24, 8))
    train_predict = model.predict(train_X)
    valid_predict = model.predict(valid_X)
    test_predict = model.predict(test_X)
    train_predict = train_predict.reshape(train_predict.shape[0], )
    valid_predict = valid_predict.reshape(valid_predict.shape[0], )
    test_predict = test_predict.reshape(test_predict.shape[0], )
    # print(type(valid_predict), type(test_predict))
    # print(valid_predict.shape, test_predict.shape)
    # plt.plot(values[:, -1], c='b')
    # plt.plot([x for x in train_predict], c='g')
    # plt.plot([None for _ in train_predict] + [x for x in valid_predict], c='y')
    # plt.plot([None for _ in train_predict] + [None for _ in valid_predict] + [x for x in test_predict], c='r')
    acc = model_acc(test_predict, test_y, error)
    print(acc)
    return acc

def model_run2(file_path,error):
    # CNN模型
    # dataset_pre
    X, y, data1 = obtain_dataset(file_path)
    train_X, test_X, train_y, test_y = seg_dataset(X, y, test_size=0.2)
    # train_X, test_X, train_y, test_y = X, X, y, y
    print(test_y[0:10])
    print(type(train_X), train_X.shape,train_y.shape)
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    # config
    n_hidden_1 = 256  # 设定隐藏层
    n_classes = 1  # 设定最后的输出层
    training_epochs = 15  # 设定整体训练数据共训练多少次
    batch_size = 100  # 设定每次提取多少样本

    # cnn_model_init
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, padding='same', input_shape=(2, 1), activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(n_hidden_1, activation='relu'))
    model.add(Dense(n_classes, activation=None))
    model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics=['accuracy'])
    # fit network
    cnn = model.fit(train_X, train_y, batch_size=batch_size, epochs=training_epochs, validation_data=(test_X, test_y))

    # plot history
    # plt.plot(cnn.history['loss'], label='train')
    # plt.plot(cnn.history['val_loss'], label='valid')
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=(24, 8))
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)
    train_predict = train_predict.reshape(train_predict.shape[0], )
    test_predict = test_predict.reshape(test_predict.shape[0], )
    # print(test_predict[0:10],test_y[0:10])
    # print(type(test_predict))
    # print(test_predict.shape)
    # plt.plot(y, c='b')
    # plt.plot([x for x in train_predict], c='g')
    # plt.plot([None for _ in train_predict], c='y')
    # plt.plot([None for _ in train_predict]+ [x for x in test_predict], c='r')
    acc = model_acc(test_predict,test_y,error)
    print(acc)
    return acc

def Model(s):
    if s is 'MLP':
        return MLPRegressor()
    elif s is 'tree':
        return tree.DecisionTreeRegressor()
    elif s is 'PassiveAggressive':
        return PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)
    elif s is 'SGDRegressor':
        return SGDRegressor(max_iter=1000, tol=1e-3)
    elif s is 'Ridgeregression':
        return linear_model.Ridge(alpha=.5)
    elif s is 'BayesianRidge':
        print('jfkjf')
        return linear_model.BayesianRidge()
    else:
        print('该模型还未实现，尽请期待！！')

def model_select(name,file_path,error):
    if name is "LSTM":
        acc = model_run1(file_path,error)
    elif name is "CNN":
        acc = model_run2(file_path,error)
    else:
        acc = model_run(name,file_path,error)
    return acc

if __name__ == '__main__':
    file_path = '../data/brake_only.csv'
    names = ['MLP','LSTM','CNN']
    names = ['MLP','tree','PassiveAggressive','SGDRegressor','Ridgeregression','BayesianRidge','LSTM','CNN']
    errors = [0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ans = {}
    for name, error in zip(names,errors):
        print(name)
        acc = model_select(name, file_path, error)
        ans[name] = '使用模型:  {0},   在设置绝对误差为 {2} 情况下，    acc为: {1}'.format(name,acc,error)
    for k, value in ans.items():
        print(value)

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # print(X_train[0:5],X_test[0:5])
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(X_train[0:5], X_test[0:5])
import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import datetime,time
import MyConfig
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2


def sub_sample(x, y,win_len):
    new_x = []
    new_y = []

    for idx, each in enumerate(x):
        original_len = each.shape[0]
        for i in range(0, original_len - win_len + 1, 1):
            x_win = each[i:i+win_len]
            new_x.append(x_win)
            new_y.append(y[idx])

    return new_x,np.array(new_y)

def split_sub_sample(x, y,win_len):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for idx, each in enumerate(x):

        original_len = each.shape[0]
        split_point = original_len*4/5
        train = each[:split_point]
        test = each[split_point:]

        for i in range(0, train.shape[0] - win_len + 1, 1):
            x_win = train[i:i+win_len]
            X_train.append(x_win)
            y_train.append(y[idx])
        for i in range(0, test.shape[0] - win_len + 1, 1):
            x_win = test[i:i+win_len]
            X_test.append(x_win)
            y_test.append(y[idx])

    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

def get_model():
    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, dropout_W=0.5, dropout_U=0.5, input_dim=X_train.shape[2],
                   input_length=X_train.shape[1], return_sequences=True,
                   W_regularizer=l2(0.01), U_regularizer=l2(0.01)))
    model.add(LSTM(64, dropout_W=0.5, dropout_U=0.5, W_regularizer=l2(0.01), U_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    x,y = joblib.load('xiao_dataset_act_cal_scr_lig.pkl')
    #x = [each[0] for each in x]
    y = np.array(y)

    win_len = 10
    batch_size = 16

    cross_subject = True
    if cross_subject:
        #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state = 2)
        loo = LeaveOneOut()
        accs = []
        for train_idx, test_idx in loo.split(x):
            #X_train, X_test = x[train_idx], x[test_idx]
            X_train = [x[i] for i in train_idx]
            X_test = [x[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train,y_train = sub_sample(X_train,y_train,win_len = win_len)
            X_test, y_test = sub_sample(X_test, y_test,win_len = win_len)
            X_train = sequence.pad_sequences(X_train, maxlen=win_len)
            X_test = sequence.pad_sequences(X_test, maxlen=win_len)
            X_train = np.array(X_train)
            X_test  = np.array(X_test)

            model = get_model()
            print('Train...')
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100,
                      validation_data=(X_test, y_test))
            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=batch_size)
            print('')
            print('Test score:', score)
            print('Test accuracy:', acc)
            accs.append(acc)

        print('Leave one out accuracy:',np.mean(accs))

    else:
        X_train, X_test, y_train, y_test = split_sub_sample(x, y, win_len=win_len)
        X_train = sequence.pad_sequences(X_train, maxlen=win_len)
        X_test = sequence.pad_sequences(X_test, maxlen=win_len)









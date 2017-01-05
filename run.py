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
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Embedding,Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_activations(model, layer, X_batch):
    outputs = model.layers[layer].output
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [outputs])
    activations = get_activations([X_batch,0])
    return activations

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
    # model.add(LSTM(64, dropout_W=0.5, dropout_U=0.5,  return_sequences=True,
    #                W_regularizer=l2(0.01), U_regularizer=l2(0.01), input_shape=(X_train.shape[1],X_train.shape[2])))
    # model.add(LSTM(64, dropout_W=0.5, dropout_U=0.5, W_regularizer=l2(0.01), U_regularizer=l2(0.01)))
    model.add(Dense(64,activation='relu',W_regularizer=l2(0.01),input_dim=X_train.shape[1]))
    model.add(Dropout(p=0.5))
    model.add(Dense(64, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(p=0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    x,y = joblib.load('xiao_dataset.pkl')


    #x = [each[0] for each in x]
    y = np.array(y)

    win_len = 10
    batch_size = 16

    cross_subject = True
    if cross_subject:
        #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state = 2)
        loo = KFold(n_splits=10)
        accs = []
        for idx,(train_idx, test_idx) in enumerate(loo.split(x)):
            #X_train, X_test = x[train_idx], x[test_idx]
            X_train = [x[i] for i in train_idx]
            X_test = [x[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train,y_train = sub_sample(X_train,y_train,win_len = win_len)
            X_test, y_test = sub_sample(X_test, y_test,win_len = win_len)
            X_train = sequence.pad_sequences(X_train, maxlen=win_len,dtype='float')
            X_test = sequence.pad_sequences(X_test, maxlen=win_len, dtype='float')
            X_train = np.array(X_train)
            X_test  = np.array(X_test)

            f = np.array([ [0,  1,  2,  3,  4,  5, 6],
                           [7,  8,  9, 10, 11, 12, 13],
                          [14, 15, 16, 17, 18, 19, 20],
                          [21, 22, 23, 24, 25, 26, 27]   ])
            f = f.flatten()
            X_train = X_train[:, :, f]
            X_test = X_test[:, :, f]
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
            X_test = np.reshape(X_test, (X_test.shape[0], -1))
            model = get_model()
            model.summary()
            print('Train...')
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20,
                      validation_data=(X_test, y_test))
            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=batch_size)

            # plot feature maps
            pca = PCA(n_components=3)
            my_featuremaps = get_activations(model, 2, X_test)[0]
            x_projected = pca.fit_transform(my_featuremaps)
            #
            x_underpessed =  np.array([ each[0] for each in zip(x_projected, y_test) if each[1]== 0])
            x_derpessed = np.array([each[0] for each in zip(x_projected, y_test) if each[1] == 1] )
            #
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_underpessed[:,0], x_underpessed[:,1], x_underpessed[:,2], c='g', marker='o')
            ax.scatter(x_derpessed[:, 0], x_derpessed[:, 1], x_derpessed[:, 2], c='r', marker='o')
            plt.show()

            #confusion matrix
            # y_pred = model.predict(X_test)
            # y_pred = y_pred.reshape((y_pred.shape[0],))
            # y_pred[y_pred>0.5] = 1
            # y_pred[y_pred<=0.5] = 0
            # cnf_matrix = confusion_matrix(y_test, y_pred)
            # plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            # plt.title('Confusion Matrix')
            # plt.colorbar()
            # plt.show()




            print('')
            print('Test score:', score)
            print('Test accuracy:', acc)
            accs.append(acc)
            print(idx,'Mean accuracy:', np.mean(accs))
        print('Final Mean accuracy:',np.mean(accs))

    else:
        X_train, X_test, y_train, y_test = split_sub_sample(x, y, win_len=win_len)
        X_train = sequence.pad_sequences(X_train, maxlen=win_len,dtype='float')
        X_test = sequence.pad_sequences(X_test, maxlen=win_len,dtype='float')
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        f = np.array([0, 1, 2, 3, 4, 5, 6,
                      7, 8, 9, 10, 11, 12 , 13,
                      14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26 ,27])
        #f = f[:, 5]

        X_train = X_train[:, :, f]
        X_test = X_test[:, :, f]
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        model = get_model()
        print('Train...')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50,
                  validation_data=(X_test, y_test))
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size)
        print('')
        print('Test score:', score)
        print('Test accuracy:', acc)









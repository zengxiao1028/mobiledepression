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
from keras.layers import Dense, Dropout, Activation, Embedding,Bidirectional, Reshape
from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.layers import Convolution2D, MaxPooling2D, Flatten
import keras
from keras.optimizers import RMSprop

def get_activations(model, layer, X_batch):
    outputs = model.layers[layer].output
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [outputs])
    activations = get_activations([X_batch,0])
    return activations

def sub_sample(x, y,win_len):

    new_x_mean = []
    new_y = []

    new_x_std = []
    for idx, subject in enumerate(x):
        original_len = subject.shape[0]
        for i in range(0, original_len - win_len + 1, 1):
            x_win = subject[i:i+win_len]

            weekdays = np.array([each[1:] for each in x_win if 1 <= each[0] <= 5])
            weekends = np.array([each[1:] for each in x_win if 6 <= each[0] <= 7])
            #there are 10 weekdays and 4 weekends
            assert weekdays.shape[0]==10
            assert weekends.shape[0]==4

            weekdays = np.reshape(weekdays,(weekdays.shape[0], 24, -1))
            weekends = np.reshape(weekends, (weekends.shape[0], 24, -1))

            #stat mean
            weekdays_mean = np.mean(weekdays,axis=0)
            weekends_mean = np.mean(weekends,axis=0)
            stat_mean = np.array([weekdays_mean,weekends_mean])
            new_x_mean.append(stat_mean)
            new_y.append(y[idx])

            # stat std
            weekdays_std = np.std(weekdays, axis=0)
            weekends_std = np.std(weekends, axis=0)
            stat_std = np.array([weekdays_std, weekends_std])

            new_x_std.append(stat_std)

    return np.array(new_x_mean),np.array(new_y)


def split_sub_sample(x, y,win_len):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for idx, each in enumerate(x):

        original_len = each.shape[0]
        split_point = original_len*5/10
        train = each[:split_point]
        test = each[split_point:]

        for i in range(0, train.shape[0] - win_len + 1, 1):
            x_win = train[i:i+win_len]

            weekdays = np.array([each[1:] for each in x_win if 1 <= each[0] <= 5])
            weekends = np.array([each[1:] for each in x_win if 6 <= each[0] <= 7])
            # there are 10 weekdays and 4 weekends
            assert weekdays.shape[0] == 10
            assert weekends.shape[0] == 4

            weekdays = np.reshape(weekdays, (weekdays.shape[0], 24, -1))
            weekends = np.reshape(weekends, (weekends.shape[0], 24, -1))

            # stat mean
            weekdays_mean = np.mean(weekdays, axis=0)
            weekends_mean = np.mean(weekends, axis=0)
            stat_mean = np.array([weekdays_mean, weekends_mean])
            #
            # # stat std
            # weekdays_std = np.std(weekdays, axis=0)
            # weekends_std = np.std(weekends, axis=0)
            # stat_std = np.array([weekdays_std, weekends_std])
            #
            # #stat_mean_std = np.concatenate((stat_mean,stat_std),axis=2)

            X_train.append(stat_mean)
            y_train.append(y[idx])

        for i in range(0, test.shape[0] - win_len + 1, 1):
            x_win = test[i:i+win_len]

            weekdays = np.array([each[1:] for each in x_win if 1 <= each[0] <= 5])
            weekends = np.array([each[1:] for each in x_win if 6 <= each[0] <= 7])
            # there are 10 weekdays and 4 weekends
            assert weekdays.shape[0] == 10
            assert weekends.shape[0] == 4

            weekdays = np.reshape(weekdays, (weekdays.shape[0], 24, -1))
            weekends = np.reshape(weekends, (weekends.shape[0], 24, -1))

            # stat mean
            weekdays_mean = np.mean(weekdays, axis=0)
            weekends_mean = np.mean(weekends, axis=0)
            stat_mean = np.array([weekdays_mean, weekends_mean])

            # stat std
            weekdays_std = np.std(weekdays, axis=0)
            weekends_std = np.std(weekends, axis=0)
            stat_std = np.array([weekdays_std, weekends_std])

            #stat_mean_std = np.concatenate((stat_mean, stat_std), axis=2)

            X_test.append(stat_mean)
            y_test.append(y[idx])

    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)



def get_model(X_train):
    print('Build model...')
    model = Sequential()

    model.add(Convolution2D(128, 1, 5, W_regularizer=l2(0.0001),input_shape=X_train.shape[1:] ,activation='relu'))
    model.add( MaxPooling2D(pool_size=(1, 3),strides=(1,1) ) )
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 2, 3, W_regularizer=l2(0.0001),activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128,W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-08, decay=0)
    #optimizer = keras.optimizers.Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# def get_model(X_train):
#     print('Build model...')
#     model = Sequential()
#
#     # model.add(Flatten(input_shape=X_train.shape[1:]))
#     # model.add(Dense(128,activation='relu',W_regularizer=l2(0.001)))
#     # model.add(Dropout(p=0.5))
#     # model.add(Dense(128, activation='relu', W_regularizer=l2(0.001)))
#     # model.add(Dropout(p=0.5))
#
#     model.add(Convolution2D(128,1, 5, W_regularizer=l2(0.001),input_shape=X_train.shape[1:] ,activation='relu'))
#     model.add( MaxPooling2D(pool_size=(1, 3),strides=(1,1) ) )
#     model.add(Dropout(0.5))
#
#     model.add(Convolution2D(128, 1, 3, W_regularizer=l2(0.001),activation='relu'))
#
#     model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(64,W_regularizer=l2(0.01)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#
#
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     model.summary()
#     return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    x,y = joblib.load('../data_prepare/xiao_dataset.pkl')

    y = np.array(y)

    win_len = 14
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


            model = get_model(X_train)
            earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
            print('Train...')
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20,
                      callbacks=[earlyStopping],
                      validation_split=0.1,
                      #validation_data=(X_test,y_test)
                      )

            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=batch_size)

            # plot feature maps
            # pca = PCA(n_components=3)
            # my_featuremaps = get_activations(model, 2, X_test)[0]
            # x_projected = pca.fit_transform(my_featuremaps)
            # #
            # x_underpessed =  np.array([ each[0] for each in zip(x_projected, y_test) if each[1]== 0])
            # x_derpessed = np.array([each[0] for each in zip(x_projected, y_test) if each[1] == 1] )
            # #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(x_underpessed[:,0], x_underpessed[:,1], x_underpessed[:,2], c='g', marker='o')
            # ax.scatter(x_derpessed[:, 0], x_derpessed[:, 1], x_derpessed[:, 2], c='r', marker='o')
            # plt.show()

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



        # X_train = np.reshape(X_train, (X_train.shape[0], -1))
        # X_test = np.reshape(X_test, (X_test.shape[0], -1))

        model = get_model(X_train)
        print('Train...')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50,
                  validation_data=(X_test, y_test))
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size)
        print('')
        print('Test score:', score)
        print('Test accuracy:', acc)









import os
import pickle

import numpy as np

import datetime,time
import project_config
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from keras.preprocessing import sequence
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

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
    names = ["Random Forest","AdaBoost","GB"]
    classifiers = [
        RandomForestClassifier(n_estimators=200),
        AdaBoostClassifier(),
        GradientBoostingClassifier()]

    return names,classifiers


if __name__ == '__main__':

    # x,y = joblib.load('xiao_dataset_act.pkl')
    # x = [each[0] for each in x]
    # y = np.array(y)

    x, y = joblib.load('../data_prepare/xiao_dataset.pkl')
    y = np.array(y)


    win_len = 7
    batch_size = 16

    cross_subject = True
    if cross_subject:
        names, classifiers = get_model()
        for name, classifier in zip(names, classifiers):
            print('Using'+name )

            loo = KFold(n_splits=5)
            accs = []

            print('Train...')
            for idx,(train_idx, test_idx) in enumerate(loo.split(x)):
                #X_train, X_test = x[train_idx], x[test_idx]
                X_train = [x[i] for i in train_idx]
                X_test = [x[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train,y_train = sub_sample(X_train,y_train,win_len = win_len)
                X_test, y_test = sub_sample(X_test, y_test,win_len = win_len)
                #X_train = sequence.pad_sequences(X_train, maxlen=win_len)
                #X_test = sequence.pad_sequences(X_test, maxlen=win_len)
                X_train = np.array(X_train)
                X_test  = np.array(X_test)
                X_train = np.reshape(X_train,(X_train.shape[0],-1))
                X_test = np.reshape(X_test, (X_test.shape[0], -1))
                classifier.fit(X_train, y_train)
                acc = classifier.score(X_test, y_test)
                accs.append(acc)
            print('Accuracies:',np.mean(accs))
            print('')

    else:
        names, classifiers = get_model()
        for name, classifier in zip(names, classifiers):
            print('Using '+name)

            loo = LeaveOneOut()
            accs = []

            print('Train...')

            X_train, X_test, y_train, y_test = split_sub_sample(x, y, win_len=win_len)
            X_train = sequence.pad_sequences(X_train, maxlen=win_len)
            X_test = sequence.pad_sequences(X_test, maxlen=win_len)
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
            X_test = np.reshape(X_test, (X_test.shape[0], -1))
            classifier.fit(X_train, y_train)
            acc = classifier.score(X_test, y_test)

            print('Final accuracy:',acc)
            print('')









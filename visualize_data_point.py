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
from keras.layers import Dense, Dropout, Activation, Embedding,Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy
import cv2
from sklearn.decomposition import PCA

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
        split_point = original_len*1/5
        train = each[split_point:]
        test = each[:split_point]

        for i in range(0, train.shape[0] - win_len + 1, 1):
            x_win = train[i:i+win_len]
            X_train.append(x_win)
            y_train.append(y[idx])
        for i in range(0, test.shape[0] - win_len + 1, 1):
            x_win = test[i:i+win_len]
            X_test.append(x_win)
            y_test.append(y[idx])

    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x,y = joblib.load('xiao_dataset.pkl')
    y = np.array(y)

    #
    # x, y = sub_sample(x, y, win_len=10)
    # x = np.array(x)
    # x = np.reshape(x, (x.shape[0], -1))
    #
    # pca = PCA(n_components=3)
    # x_projected = pca.fit_transform(x)
    # print pca
    #
    # x_underpessed =  np.array([ each[0] for each in zip(x_projected,y) if each[1]== 0])
    # x_derpessed = np.array([each[0] for each in zip(x_projected, y) if each[1] == 1] )
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_underpessed[:,0], x_underpessed[:,1], x_underpessed[:,2], c='g', marker='o')
    # ax.scatter(x_derpessed[:, 0], x_derpessed[:, 1], x_derpessed[:, 2], c='r', marker='o')
    # plt.show()






    for idx, subject in enumerate(x):
        print idx
        #subject = scipy.misc.imresize(subject,300,interp='nearest')
        subject = cv2.resize(subject, (0, 0), fx=12.0, fy=12.0, interpolation=cv2.INTER_AREA)
        subject = subject*255
        #cv2.imshow('hi',each)
        if y[idx] == 1:
            cv2.imwrite('./visualization/1/subject_' + str(idx) + '_.jpg', subject)
        elif y[idx]==0:
            cv2.imwrite('./visualization/0/subject_' + str(idx) + '_.jpg', subject)
        else:
            pass

    # for idx, each in enumerate(X_train):












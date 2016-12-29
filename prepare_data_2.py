import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import datetime,time
from datetime import timedelta
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from sklearn.model_selection import train_test_split

def datetime2sec(dt):
    sec = time.mktime(dt.timetuple())
    return int(sec)


def stat_day(time_slice,raw_data_dict):

    _00 = datetime2sec(time_slice - timedelta(hours=24))
    _06 = datetime2sec(time_slice - timedelta(hours=18))
    _12 = datetime2sec(time_slice - timedelta(hours=12))
    _18 = datetime2sec(time_slice - timedelta(hours=6))
    _24 = datetime2sec(time_slice)
    spans = [(_00,_06),(_06,_12),(_12,_18),(_18,_24)]

    feature = []
    for start,end in spans:
        # activity data
        data_act = raw_data_dict['act']
        ind = np.where(data_act[0].between(start, end, inclusive=True))[0]
        if ind.size:
            act_onfoot = np.sum(data_act[1][ind] == 'ON_FOOT') / float(ind.size)
            act_still = np.sum(data_act[1][ind] == 'STILL') / float(ind.size)
            act_invehicle = np.sum(data_act[1][ind] == 'IN_VEHICLE') / float(ind.size)
            act_tilting = np.sum(data_act[1][ind] == 'TILTING') / float(ind.size)
            act_confidence = np.nanmean(data_act[2][ind])
        else:
            act_onfoot = 0
            act_still = 1
            act_invehicle = 0
            act_tilting = 0
            act_confidence = 100


        span_feature = np.array([act_onfoot,act_still,act_invehicle,act_tilting,act_confidence])
        feature.append(span_feature)

    feature = np.hstack(feature)
    return feature



def prepare_data(subjects,data_dir ):

    with open('target.pkl') as f:
        filtered_labels = pickle.load(f)
    f.close()

    x = []
    y = []

    print ('total subjects:', len(subjects))
    for (s, subject) in enumerate(subjects):
        print (s,subject)
        if subject not in filtered_labels.keys():
            print ('skip subject:',subject)
            continue

        ###read data
        raw_data_dict = dict()

        if os.path.exists(data_dir + subject + '/act.csv'):
            data_act = pd.read_csv(data_dir + subject + '/act.csv', sep='\t', header=None)
            raw_data_dict.update({'act': data_act})
        else:
            print (' skipping - no data')
            continue

        ### determine time slices
        start_dt = datetime.datetime.fromtimestamp(data_act.as_matrix()[0][0])
        start_dt = start_dt.replace(hour=00, minute=0,second=1)
        print (start_dt)
        end_dt = datetime.datetime.fromtimestamp(data_act.as_matrix()[-1][0])
        end_dt = end_dt.replace(hour=00, minute=00, second = 1 ) + timedelta(days=1)
        print (end_dt)
        print
        time_slices = []
        slice = start_dt + timedelta(days=1)
        while(slice<=end_dt):
            time_slices.append(slice)
            slice = time_slices[-1] + timedelta(days=1)

        # samples for each subject
        features = []
        for time_slice in time_slices:
            #every_day is represented by a feature vector
            feature = stat_day(time_slice,raw_data_dict)
            features.append(feature)
        features = np.array(features)
        print (len(features))

        x.append(np.array(features))
        y.append(filtered_labels[subject])


    return x,y

def sub_sample(x, y):
    new_x = []
    new_y = []
    for idx, each in enumerate(x):
        x_1 = each[:len(x)/2]
        x_2 = each[len(x)/2:]
        new_x.append(x_1)
        new_x.append(x_2)
        new_y.append(y[idx])
        new_y.append(y[idx])
    return new_x,np.array(new_y)


if __name__ == '__main__':


    # subjects = os.listdir(data_dir)
    #
    # x,y = prepare_data(subjects,data_dir)
    # joblib.dump((x,y),'xiao_dataset.pkl',compress=3)

    x,y = joblib.load('xiao_dataset.pkl')
    x = [each[0] for each in x]
    y = np.array(y)
    print (len(x),len(y))

    x,y = sub_sample(x,y)
    print (x[0].shape,y[0].shape)

    maxlen = 30
    batch_size = 4

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=1)

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(LSTM(128,dropout_W=0.1, dropout_U=0.1,input_shape=(X_train.shape[1], X_train.shape[2])))  # try using a GRU instead, for fun
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=30,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    print('\nTest score:', score)
    print('Test accuracy:', acc)




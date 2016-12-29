import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import datetime,time
from datetime import timedelta
from sklearn.externals import joblib
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

        x.append(np.array([features]))
        y.append(filtered_labels[subject])


    return x,y


if __name__ == '__main__':

    data_dir = 'G:\depressiondata\CS120Data\CS120\\'
    subjects = os.listdir(data_dir)

    # x,y = prepare_data(subjects,data_dir)
    # joblib.dump((x,y),'xiao_dataset.pkl',compress=3)

    x,y = joblib.load('xiao_dataset.pkl')
    print (len(x),len(y))




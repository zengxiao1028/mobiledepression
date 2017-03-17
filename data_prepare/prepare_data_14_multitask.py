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


def datetime2sec(dt):
    sec = time.mktime(dt.timetuple())
    return int(sec)


def stat_day(time_slice,raw_data_dict):

    hours_dt = dict()
    interval = 24

    # for x in range(0, 25, interval):
    #     hours_dt[x] = datetime2sec(time_slice - timedelta(hours=(24-x)))
    # spans = []
    # for x in range(0, 24, interval):
    #     spans.append((hours_dt[x],hours_dt[x+interval]))

    _00 = datetime2sec(time_slice - timedelta(hours=24))
    _06 = datetime2sec(time_slice - timedelta(hours=18))
    _12 = datetime2sec(time_slice - timedelta(hours=12))
    _18 = datetime2sec(time_slice - timedelta(hours=6))
    _24 = datetime2sec(time_slice)
    spans = [(_00,_06)]

    feature = []
    for idx,(start,end) in enumerate(spans):

        # screen data
        data_scr = raw_data_dict['scr']
        ind = np.where(data_scr[0].between(start, end, inclusive=True))[0]
        if ind.size:
            changes_time = data_scr[0][ind]
            scr_changes = data_scr[1][ind]
            begin = -1
            is_recording_on = False
            total_on_time = 0
            for time,is_turn_on in zip(changes_time,scr_changes):
                #screen from off to on
                if is_turn_on:
                    begin = time
                    is_recording_on = True
                #screen from on to off
                else:
                    if is_recording_on:
                        on_time = time - begin
                        total_on_time += on_time
                    is_recording_on = False

            total_time = data_scr[0][ind[-1]] - data_scr[0][ind[0]]
            scr_n = total_on_time*1.0 / total_time

        else:
            scr_n = 0


        span_feature = np.array(scr_n)

        feature.append(span_feature)

    feature = np.hstack(feature)
    return np.array(feature)



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

        if os.path.exists(data_dir + subject + '/cal.csv'):
            data_cal = pd.read_csv(data_dir + subject + '/cal.csv', sep='\t', header=None)
            raw_data_dict.update({'cal': data_cal})
        else:
            print ' skipping - no data'
            continue

        if os.path.exists(data_dir + subject + '/scr.csv'):
            data_scr = pd.read_csv(data_dir + subject + '/scr.csv', sep='\t', header=None)
            raw_data_dict.update({'scr': data_scr})
        else:
            print ' skipping - no data'
            continue

        if os.path.exists(data_dir + subject + '/lgt.csv'):
            data_lgt = pd.read_csv(data_dir + subject + '/lgt.csv', sep='\t', header=None)
            raw_data_dict.update({'lgt': data_lgt})
        else:
            print ' skipping - no data'
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

        x.append( np.array(features) )
        y.append( filtered_labels[subject] )

    print 'remained subjects:',len(x)
    return x,y






if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    subjects = os.listdir(MyConfig.data_dir)

    x,y = prepare_data(subjects,MyConfig.data_dir)
    joblib.dump((x,y),'xiao_dataset_raw.pkl',compress=3)





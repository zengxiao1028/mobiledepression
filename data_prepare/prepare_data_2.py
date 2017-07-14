import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import datetime,time
import project_config
from datetime import timedelta
from sklearn.externals import joblib
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
    for idx,(start,end) in enumerate(spans):
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

        # call data
        data_cal = raw_data_dict['cal']
        ind = np.where(data_cal[0].between(start, end, inclusive=True))[0]
        if ind.size:
            cal_dur = np.sum(data_cal[1][ind] == 'Off-Hook') / float(ind.size)
        else:
            cal_dur = 0

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

            total_time = end - start
            scr_n = total_on_time*1.0 / total_time

        else:
            scr_n = 0

        # light data
        data_lgt = raw_data_dict['lgt']
        ind = np.where(data_lgt[0].between(start, end, inclusive=True))[0]
        if ind.size:
            off_idx = np.where(data_lgt[1][ind].between(0., 50., inclusive=True))[0]
            lgt_off = off_idx.size / float(ind.size)
        else:
            lgt_off = 0

        # if idx == 0:
        #     span_feature = np.array([act_onfoot,act_still,act_invehicle,act_tilting,cal_dur,scr_n,lgt_off])
        # else:
        #     span_feature = np.array([act_onfoot, act_still, act_invehicle, act_tilting, cal_dur, scr_n])

        span_feature = np.array([act_onfoot, act_still, act_invehicle, act_tilting, cal_dur, scr_n, lgt_off])
        feature.append(span_feature)

    feature = np.hstack(feature)
    return feature



def prepare_data(subjects, data_dir ):

    # with open('target.pkl') as f:
    #     filtered_labels = pickle.load(f)
    # f.close()


    x = []
    y = []
    print ('total subjects:', len(subjects))
    for (s, subject) in enumerate(subjects):
        print (s,subject)

        # if subject not in filtered_labels.keys():
        #     print ('skip subject:',subject)
        #     continue

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
            print (' skipping - no data')
            continue

        if os.path.exists(data_dir + subject + '/scr.csv'):
            data_scr = pd.read_csv(data_dir + subject + '/scr.csv', sep='\t', header=None)
            raw_data_dict.update({'scr': data_scr})
        else:
            print (' skipping - no data')
            continue

        if os.path.exists(data_dir + subject + '/lgt.csv'):
            data_lgt = pd.read_csv(data_dir + subject + '/lgt.csv', sep='\t', header=None)
            raw_data_dict.update({'lgt': data_lgt})
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
        print (end_dt - start_dt)
        time_slices = []
        slice = start_dt + timedelta(days=1)
        while(slice <= end_dt):
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
        #y.append( filtered_labels[subject] )

    print ('remained subjects:',len(x))
    #return x,y
    return x,0






if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    subjects = os.listdir(project_config.DATA_DIR)

    x,y = prepare_data(subjects,project_config.DATA_DIR)
    joblib.dump((x,y),'xiao_dataset.pkl',compress=3)





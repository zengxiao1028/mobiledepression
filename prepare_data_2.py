# coding: utf-8

# In[7]:

# This code prepares sensor and EMA data for the LSTM

def prepare_data(subjects,data_dir ):
    import pandas as pd
    import numpy as np
    from scipy import stats
    import os

    deltat = 1800
    win = 24 * 3600  # in the past one day

    x = np.array([])
    y = np.array([])

    print 'total subjects:', len(subjects)
    for (s, subject) in enumerate(subjects):

        print s,subject
        if os.path.exists(data_dir + subject + '/act.csv'):
            data_act = pd.read_csv(data_dir + subject + '/act.csv', sep='\t', header=None)
        else:
            print ' skipping - no data'
            continue


        if os.path.exists(data_dir + subject + '/cal.csv'):
            data_cal = pd.read_csv(data_dir + subject + '/cal.csv', sep='\t', header=None)
        else:
            print ' skipping - no data'
            continue

        if os.path.exists(data_dir + subject + '/coe.csv'):
            data_coe = pd.read_csv(data_dir + subject + '/coe.csv', sep='\t', header=None)
        else:
            print ' skipping - no data'
            continue

        if os.path.exists(data_dir + subject + '/scr.csv'):
            data_scr = pd.read_csv(data_dir + subject + '/scr.csv', sep='\t', header=None)
        else:
            print ' skipping - no data'
            continue

        if os.path.exists(data_dir + subject + '/emm.csv'):
            target = pd.read_csv(data_dir + subject + '/emm.csv', sep='\t', header=None)
        else:
            print ' skipping - no data'
            continue
        print


        for (i, t1) in enumerate(target[0]):
            lat = np.nan
            lng = np.nan

            for t2 in np.arange(t1 - win, t1, deltat):



                # communication data
                ind = np.where(data_coe[0].between(t2, t2 + deltat, inclusive=True))[0]
                if ind.size:
                    b = data_coe[3][ind] == 'SMS'
                    a = data_coe[3][ind] == 'SMS' and data_coe[4][ind] == 'INCOMING'

                    sms_in = np.sum(data_coe[3][ind] == 'SMS' and data_coe[4][ind] == 'INCOMING')
                    sms_out = np.sum(data_coe[3][ind] == 'SMS' and data_coe[4][ind] == 'OUTGOING')
                    sms_miss = np.sum(data_coe[3][ind] == 'SMS' and data_coe[4][ind] == 'MISSED')

                    phone_in = np.sum(data_coe[3][ind] == 'PHONE' and data_coe[4][ind] == 'INCOMING')
                    phone_out = np.sum(data_coe[3][ind] == 'PHONE' and data_coe[4][ind] == 'OUTGOING')
                    phone_miss = np.sum(data_coe[3][ind] == 'PHONE' and data_coe[4][ind] == 'MISSED')

                    print 'sms in:',sms_in*1.0/(sms_in+sms_out+sms_miss)
                    print 'sms out:', sms_out * 1.0 / (sms_in + sms_out + sms_miss)
                    print 'sms miss:', sms_miss * 1.0 / (sms_in + sms_out + sms_miss)
                    print 'phone in:', phone_in * 1.0 / (phone_in + phone_out + phone_miss)
                    print 'phone out:', phone_out * 1.0 / (phone_in + phone_out + phone_miss)
                    print 'phone miss:', phone_miss * 1.0 / (phone_in + phone_out + phone_miss)

                else:
                    sms = 0
                    phone = 0
                    incoming = 0
                    outgoing = 0
                    missed = 0

                # activity data
                ind = np.where(data_act[0].between(t2, t2 + deltat, inclusive=True))[0]
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

                # screen data
                ind = np.where(data_scr[0].between(t2, t2 + deltat, inclusive=True))[0]
                if ind.size:
                    scr_n = ind.size / 2.0
                else:
                    scr_n = 0

                # call data
                ind = np.where(data_cal[0].between(t2, t2 + deltat, inclusive=True))[0]
                if ind.size:
                    cal_dur = np.sum(data_cal[1][ind] == 'Off-Hook')
                else:
                    cal_dur = 0

                # time
                hour = np.mod(t1, 86400) / 3600.0
                dow = np.mod(t1, 86400 * 7) / 86400.0

                # input vector
                vec = np.array(
                    [lat, lng, hour, dow, sms, phone, incoming, outgoing, missed, act_onfoot, act_still, act_invehicle,
                     act_tilting, act_confidence, scr_n, cal_dur,])

                vec = vec.reshape(1, vec.size)

                # adding to input matrix
                if t2 == t1 - win:
                    x_sample = vec
                else:
                    x_sample = np.append(x_sample, vec, axis=0)

            target_vec = np.array(
                [s, target.loc[i, 1]])  # subject ID is added- shall be used for subject-wise cross-validation
            target_vec = target_vec.reshape(1, target_vec.size)

            if x.any():
                x = np.append(x, x_sample.reshape(1, x_sample.shape[0], x_sample.shape[1]), axis=0)
                y = np.append(y, target_vec, axis=0)
            else:
                x = x_sample.reshape(1, x_sample.shape[0], x_sample.shape[1])
                y = target_vec

    return [x, y]


# In[26]:

if __name__ == '__main__':

    import os
    import pickle

    data_dir = '/data/xiao/DepressionData/CS120Data/CS120/'
    subjects = os.listdir(data_dir)
    # subjects = subjects[:12]
    data = prepare_data(subjects,data_dir)
    with open('data_lstm.dat', 'w') as file_out:
        pickle.dump(data, file_out)
    file_out.close()

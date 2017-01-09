import os
from sklearn.externals import joblib
import numpy as np


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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    x,y = joblib.load('xiao_dataset.pkl')

    undepressed = []
    depressed = []

    x, y = sub_sample(x, y,win_len=14)

    for stat,dpr in zip(x,y):
        print stat[1, 2, :]
        if dpr == 0:
            scr_on = stat[1, :, 6]
            undepressed.append(np.sum(scr_on))
        elif dpr == 1:
            scr_on = stat[1, :, 6]
            depressed.append(np.sum(scr_on))

    print np.mean(undepressed), np.std(undepressed)
    print np.mean(depressed), np.std(depressed)
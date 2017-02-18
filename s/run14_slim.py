import os
import pickle
import tensorflow as tf
import numpy as np
from scipy import stats
import datetime,time
import MyConfig
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from depression_net import depress_net

def sub_sample(x, y,win_len):

    new_x_mean = []
    new_y = []

    new_x_std = []

    # for each subject
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

    #for each subject
    for idx, each in enumerate(x):

        original_len = each.shape[0]
        split_point = original_len*5/10
        train = each[:split_point]
        test = each[split_point:]

        for i in range(0, train.shape[0] - win_len + 1):
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

def gen_data(x, y , b_size) :
    x, y = shuffle(x, y)
    k = x.shape[0] / b_size
    for i in range(k):
        x_batch = x[i*b_size:b_size*(i+1)]
        y_batch = y[i*b_size:b_size*(i+1)]
        yield x_batch,y_batch


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    x,y = joblib.load('../data_prepare/xiao_dataset.pkl')
    y = np.expand_dims(np.array(y), axis=1)

    win_len = 14

    cross_subject = True

    if cross_subject:
        loo = KFold(n_splits=10,random_state=1024)
        accs_mean = []
        net = None
        for idx,(train_idx, test_idx) in enumerate(loo.split(x)):
            #X_train, X_test = x[train_idx], x[test_idx]
            X_train = [x[i] for i in train_idx]
            X_test = [x[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train,y_train = sub_sample(X_train,y_train,win_len = win_len)
            X_test, y_test = sub_sample(X_test, y_test,win_len = win_len)
            if net is None:
                net = depress_net(X_train.shape)
            else:
                pass
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                num_epoch = 5
                while num_epoch>0:
                    num_epoch -= 1
                    for X_batch, y_batch in gen_data(X_train, y_train, 16):
                        _, loss, acc, step = sess.run([net.optimizor,net.loss,net.acc,net.global_step],
                            feed_dict={net.x_ph:X_batch,net.y_ph:y_batch, net.is_training:True })



                        if step%10==0:
                            loss, acc = sess.run([net.loss, net.acc], feed_dict={net.x_ph: X_test,
                                                            net.y_ph: y_test,
                                                            net.is_training: False})
                            print('test %d, loss %f, acc %f' % (step, loss, acc))
                test_acc = sess.run([ net.acc],feed_dict={net.x_ph: X_test,
                                                         net.y_ph: y_test,
                                                         net.is_training: False})
            accs_mean.append(test_acc)


        print np.array(accs_mean).mean()






    else:
        X_train, X_test, y_train, y_test = split_sub_sample(x, y, win_len=win_len)



        # X_train = np.reshape(X_train, (X_train.shape[0], -1))
        # X_test = np.reshape(X_test, (X_test.shape[0], -1))

        model = get_network(X_train)
        print('Train...')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50,
                  validation_data=(X_test, y_test))
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size)
        print('')
        print('Test score:', score)
        print('Test accuracy:', acc)









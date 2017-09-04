import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import datetime,time
import project_config
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.model_selection import KFold, LeaveOneOut

from sklearn.metrics import mean_squared_error
from paper_experiment.keras_models import lstm_model,cnn_model,mlp_model
from keras.callbacks import EarlyStopping
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



if __name__ == '__main__':
    ##prepared_label.py -> classification_labels.pkl
    ##classification_labels.pkl -> prepared_data_14.py -> quarter_feature_classification.pkl
    # quarter_feature_classification.pkl,classification_labels.pkl -> combine_1day_feature -> quarter.pkl

    x, y = joblib.load('./quarter.pkl')
    y = np.array(y)

    win_len = 7
    batch_size = 16


    #loo = KFold(n_splits=5)
    loo = LeaveOneOut()
    accs = []
    residuals = []
    y_tests = []
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
        # X_train = np.reshape(X_train,(X_train.shape[0],-1))
        # X_test = np.reshape(X_test, (X_test.shape[0], -1))

        print('Start training...')

        model = cnn_model(X_train.shape)
        earlyStopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
        model.fit(X_train, y_train, batch_size=batch_size, epochs=20,
                  callbacks=[earlyStopping],
                  validation_split=0.1)


        print('Start predicting...')
        # predict
        model.fit(X_train, y_train, batch_size=batch_size, epochs=20,
                  callbacks=[earlyStopping],
                  validation_split=0.1)
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=batch_size)
        accs.append(acc)
    print('Final Accuracies:',np.mean(accs))











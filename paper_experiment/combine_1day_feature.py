import numpy as np
import scipy.io as sio
import project_config
import os
import pickle
import datetime
import time
import calendar
import pickle
from sklearn.externals import joblib
def datetime2sec(dt):
    sec = time.mktime(dt.timetuple())
    return int(sec)

def main(feature_path):

    #with open('1day_feature.pkl', 'rb') as f:
    #    results = pickle.load(f)

    results,_ = joblib.load(feature_path)

    result=dict()

    file = os.path.join(project_config.PROJECT_ROOT_DIR, 'data_prepare/features_gps_1day_all.mat')
    gps = sio.loadmat(file)

    for idx,each in enumerate(gps['feature']):
        subject = gps['subject_feature'][0][idx][0]
        if subject in results:
            old_fea = results[subject]
            new_fea = (each[0],old_fea)
            results[subject] = new_fea

    for key,each in results.items():
        gps_features = each[0]
        other_features = each[1]
        while( gps_features[0][-1] != calendar.timegm( other_features[0][0].timetuple() ) ):

            if (gps_features[0][-1] < calendar.timegm(other_features[0][0].timetuple())):

                print(datetime.datetime.fromtimestamp(gps_features[0][-1]),
                      datetime.datetime.fromtimestamp(calendar.timegm(other_features[0][0].timetuple())))
                gps_features = gps_features[1:]
            if (gps_features[0][-1] > calendar.timegm(other_features[0][0].timetuple())):
                other_features = other_features[1:]


        while( gps_features[-1][-1] != calendar.timegm( other_features[-1][0].timetuple() ) ):

            if (gps_features[-1][-1] < calendar.timegm(other_features[-1][0].timetuple())):
                other_features = other_features[:-1]
            if (gps_features[-1][-1] > calendar.timegm(other_features[-1][0].timetuple())):
                gps_features = gps_features[:-1]



        # a = gps_features[0][-1]
        # b = datetime.datetime.utcfromtimestamp(a)
        #
        # c = calendar.timegm(other_features[0][0].timetuple())
        # d = other_features[0][0]

        # print(a,   #gps feature
        #       b,
        #       c,    #other feature
        #       d)

        #new_fea = np.hstack([gps_features[:,:-1],other_features[:,1:] ])

        #clean gps features
        gps_features_index = list(np.arange(0,gps_features.shape[1]))
        index_to_remove = [2, 15]
        for i in index_to_remove:
            gps_features_index.remove(i)
        gps_features = np.nan_to_num(gps_features[:, gps_features_index])


        new_fea = np.hstack([gps_features, other_features[:, 1:]])  # omit the first column in other features ( timestamp)
        print(new_fea[:][:].shape)

        result[key] = new_fea

    return result

if __name__ == '__main__':

    result = main('quarter_feature_classification.pkl')

    labels = joblib.load('classification_labels.pkl')
    xs = []
    ys = []
    for subject in labels.keys():
        if subject in result.keys():
            xs.append( result[subject])
            ys.append( labels[subject])

    joblib.dump((xs,ys),'quarter.pkl')
    print(len(xs))






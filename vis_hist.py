from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    scores_dict = joblib.load('scores_dict.pkl')
    x = np.array(scores_dict.values(),dtype='int32')

    x[x>=999] = -1

    plt.subplot(1, 3, 1)
    x_0 = x[:, 0]
    n, bins, patches = plt.hist(x_0,5,facecolor='green')
    plt.title('Week 0 Distribution')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    x_1 = x[:, 1]
    plt.title('Week 3 Distribution')
    n, bins, patches = plt.hist(x_1, 5, facecolor='green')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    x_2 = x[:, 2]
    plt.title('Week 6 Distribution')
    n, bins, patches = plt.hist(x_2, 5, facecolor='green')
    plt.grid(True)

    plt.show()
    print 'Finish'
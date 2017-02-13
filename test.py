import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
def gen_data(x, b_size) :
    x = shuffle(x)
    k = x.shape[0] / b_size
    for i in range(k):
        x_batch = x[i*b_size:b_size*(i+1)]
        yield x_batch

if __name__ == '__main__':

    a = np.arange(0,10)
    for each in gen_data(a,3):
        print each
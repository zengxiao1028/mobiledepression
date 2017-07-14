import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time
import datetime

def datetime2sec(dt):
    sec = time.mktime(dt.utctimetuple())
    return int(sec)

def gen_data(x, b_size) :
    x = shuffle(x)
    k = x.shape[0] / b_size
    for i in range(k):
        x_batch = x[i*b_size:b_size*(i+1)]
        yield x_batch

if __name__ == '__main__':
    utc_dt = datetime.datetime.utcfromtimestamp(1490363068)
    print(utc_dt)

    utc_sec = datetime2sec(utc_dt)
    print(utc_sec)
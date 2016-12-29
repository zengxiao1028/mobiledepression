import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import datetime,time

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

def datetime2sec(dt):
    sec = time.mktime(dt.timetuple())
    return int(sec)
now = datetime.datetime.now()
print (now)
print (datetime2sec(now))
print (datetime.datetime.fromtimestamp(datetime2sec(now)))
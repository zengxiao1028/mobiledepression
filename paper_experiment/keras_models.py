from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape

def lstm_model(input_shape):
    wd = 0.001
    print('Build LSTM model...')
    model = Sequential()
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5,  return_sequences=True,
                   kernel_regularizer=l2(wd), recurrent_regularizer=l2(wd), input_shape=(input_shape[1],input_shape[2])))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(wd), recurrent_regularizer=l2(wd)))
    model.add(Dense(64,activation='relu',kernel_regularizer=l2(wd),input_dim=input_shape[1]))
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(wd)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-08, decay=0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model

def cnn_model(input_shape):
    print('Build CNN model...')
    wd = 0.001

    model = Sequential()
    model.add(Reshape(1,input_shape[1],input_shape[2]),input_shape=input_shape[1:])
    model.add(Convolution2D(128, 1, 3, W_regularizer=l2(wd), activation='relu'))
    model.add(Convolution2D(128, 1, 3, W_regularizer=l2(wd), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3),strides=(1,1) ) )
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 1, 3, W_regularizer=l2(wd),activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128,W_regularizer=l2(wd)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-08, decay=0)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


def mlp_model(input_shape):
    print('Build MLP model...')
    wd = 0.001
    model = Sequential()

    model.add(Flatten(input_shape=input_shape[1:]))

    model.add(Dense(128,activation='relu',W_regularizer=l2(wd)))
    model.add(Dropout(p=0.5))

    model.add(Dense(128, activation='relu', W_regularizer=l2(wd)))
    model.add(Dropout(p=0.5))

    model.add(Dense(64,W_regularizer=l2(wd)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout , LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping


split = -80

def lstm(X,Y,split = split):        
        x_train , y_train = X[0:split] , Y[0:split]
        x_test , y_test = X[split:] , Y[split:]

        model = Sequential()
        model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.summary()

        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        model.fit(x_train, y_train, 
                validation_data=(x_test, y_test),
                epochs=100, batch_size=16,
                callbacks=[early_stop])

        pred = model.predict(x_test)

        return pred
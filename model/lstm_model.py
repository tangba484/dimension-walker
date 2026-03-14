from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import horizon,window
def build_model(feature_dim):

    model = Sequential()

    model.add(LSTM(64,
                   return_sequences=True,
                   input_shape=(window,feature_dim)))

    model.add(Dropout(0.2))

    model.add(LSTM(32))

    model.add(Dense(horizon))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model
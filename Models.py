from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Bidirectional, GlobalMaxPool1D

def Create_RNN(input_shape):
    # Building Model
    model = Sequential()

    # LSTM 1
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))

    # LSTM 2
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))

    # LSTM 3
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))

    # LSTM 4
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))

    # Connecting Layers
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

def load_or_train_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model_path = 'lstm_model_weights.h5'  # Use a consistent filename
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        model.fit(X, y, batch_size=1, epochs=10)
        model.save_weights(model_path)

    return model

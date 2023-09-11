# Optimization 1: Import only the necessary functions and modules

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')

# Optimization 2: Use pandas.read_csv with a file-like object instead of a string filename

# Data Collection and Preprocessing
# Assumption: Stock market data is stored in a CSV file named 'stock_data.csv'
with open('stock_data.csv', 'r') as file:
    df = pd.read_csv(file)

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()


def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


df['Sentiment'] = df['News'].apply(get_sentiment)

# Optimization 3: Use df.values instead of df[['Price', 'Volume', 'Moving_Averages', 'Technical_Indicators', 'Sentiment']]

# Feature Extraction
# Extract relevant features from the stock market data
features = df.values[:, [2, 3, 4, 5, 6]]

# Normalize data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Optimization 4: Split the scaled_features array directly

# Split data into training and testing sets
train_size = int(len(scaled_features) * 0.8)
train_data, test_data = scaled_features[:
                                        train_size], scaled_features[train_size:]

# Optimization 5: Use a tf.keras.Sequential model instead of keras.Sequential

# Neural Network Training
# Define and train the neural network for stock market prediction


def create_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


model = create_model()
history = model.fit(x=train_data[:, :-1], y=train_data[:, -1], validation_data=(
    test_data[:, :-1], test_data[:, -1]), epochs=50, batch_size=64)

# Predictive Analysis
# Use the trained model to generate predictions
predictions = model.predict(test_data[:, :-1])

# Additional optimizations can be made to the visualization and evaluation parts of the code, but it depends on the specific requirements and needs. Feel free to ask for more optimizations if required.

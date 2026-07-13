import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# --- Load data and model (This happens once when the script starts) ---
word_index = imdb.get_word_index()
model=load_model('simple_rnn_imbd.h5')
# --- Helper Function ---
def preprocess_text(text):
    """Converts raw text into a format the model can understand."""
    words = text.lower().split()
    max_features = 10000
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        if index is not None and index < max_features:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # 2 is the index for 'unknown' words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- Streamlit App UI ---
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input=st.text_area('Movie Review')

if st.button('classify'):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        sentiment = 'Positive' if score > 0.5 else 'Negative'

        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score}')
    else:
        st.write("Please enter a movie review first.")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define vocabulary size and sequence length
max_features = 10000  # Number of words to consider as features
max_len = 500

# --- 1. Load Data ---
# This will load the data and assign it to x_train, y_train, etc.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
print(f'Testing data shape: {x_test.shape}, Testing label shape: {y_test.shape}')

# --- 2. Decode a review (for demonstration) ---
word_index=imdb.get_word_index()
### to reverse the value  and key  palces  we have use the below list comphresion
reverse_word_index={value:key for key,value in word_index.items()}

# Correctly decode the first review. The original code was iterating over the dictionary.
# We need to iterate over a specific review, like x_train[0].
# The indices are offset by 3 because 0, 1, 2 are reserved for "padding", "start", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

# print("\n--- Decoded first review ---")
# print(decoded_review)
# print(f"Label: {y_train[0]}")
# print("-" * 28)

# --- 3. Preprocessing: Pad sequences ---
# Ensure all sequences have the same length.
x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)

# --- 4. Build the RNN model ---
print("Building a new model...")
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))  # Input/Embedding Layer
model.add(SimpleRNN(128, activation='relu'))                   # Hidden Layer
model.add(Dense(1, activation="sigmoid"))                      # Output Layer

# --- 5. Compile the model ---
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# --- 6. Train the model ---
history=model.fit(
    x_train,y_train,epochs=10,batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

# --- 7. Save the trained model ---
model_path = 'simple_rnn_imbd.h5'
model.save(model_path)
print(f"\nModel trained and saved to {model_path}")
model.summary()
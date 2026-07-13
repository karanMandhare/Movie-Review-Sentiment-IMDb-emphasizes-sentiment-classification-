import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.models import  load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}



model=load_model('simple_rnn_imbd.h5')
# print(model.summary())
# print(model.get_weights())



## step 2 : Helper Functions
## function to decode review 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## to preprocess the user input we need this 
def preprocess_text(text):
    words=text.lower().split()
    # The model was trained with max_features = 10000. We must respect this limit.
    max_features = 10000
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        # Check if the word is in the dictionary AND within the model's vocabulary size.
        if index is not None and index < max_features:
            encoded_review.append(index + 3)  # Add 3 to account for reserved indices 0, 1, 2.
        else:
            encoded_review.append(2)  # Use 2 for out-of-vocabulary or unknown words.
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review 


### prediction function 

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    return sentiment,prediction[0][0]


example_review="Visually spectacular but emotionally hollow. Director Marcus Vance spends so much time constructing breathtaking planetary rings and hyper-realistic alien architecture that he completely forgets to give his human characters a pulse. The third-act twist is visible from a mile away, and despite a valiant effort by the lead actress to inject some genuine dread into the script, the film ultimately succumbs to its own massive budget. Go for the eye candy, but don't expect to remember it by morning."

# Correctly unpack the returned tuple: (sentiment, prediction_score)
sentiment, prediction_score = predict_sentiment(example_review)

print(f"Review :{example_review}")
print(f"Sentiment : {sentiment}")
# Print the numerical score, not the sentiment string again
print(f"Prediction Score :{prediction_score}")
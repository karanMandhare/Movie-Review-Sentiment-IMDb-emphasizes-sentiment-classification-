from tensorflow.keras.preprocessing.text import one_hot

sent = [
    "A brilliant blue jay carefully inspected the shiny silver coin resting on the mossy stone deep within the quiet forest",
    "The vintage grandfather clock in the hallway suddenly chimed thirteen times",
    "A curious sea otter swam backward while juggling three smooth pebbles",
    "The smell of fresh cinnamon rolls drifted through the open kitchen window",
    "He misplaced his favorite green umbrella on a perfectly cloudless afternoon",
    "An ancient leather-bound book sat undisturbed on the highest shelf of the library",
    "Bright neon lights reflected off the wet pavement after the midnight thunderstorm",
    "The astronaut watched the distant blue planet rise slowly over the gray lunar horizon"
]

## define the vocabulary size 

voc_size=10000

## one hot representation 
one_hot_repr=[one_hot(words,voc_size)for words in sent]
# print(one_hot_repr)


## word embedding representation

from tensorflow.keras.layers import Embedding
# from tensorflow.keras.processing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np 

sent_length=20                          
embeded_docs=pad_sequences(one_hot_repr,padding='pre' ,maxlen=sent_length)
 ## can also use here padding='post' 

# print(embeded_docs)

## features representation
dim=10
model=Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam','mse')

# print(model.summary())

# to see what  embeddding layer is giving for each sentence 
print(embeded_docs[0])
print(model.predict(embeded_docs[0:1]))

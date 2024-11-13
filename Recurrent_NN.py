import keras
from keras import datasets
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing import sequence
#import as a function from keras module, not as a called method
from keras_preprocessing.text import text_to_word_sequence

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

### LOAD DATA
(train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words = VOCAB_SIZE)
#print(train_data[0])

### PREPROCESS
# Make review samples length be the same
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

if not os.path.exists('Recurrent_NN_1.keras'):
    ### BUILD THE MODEL
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32), # outputs 32 tensor/vector dim. for every input word in the sequence
        tf.keras.layers.LSTM(32), # 32 nodes, it gives a higher capacity to contextualize previous words
        tf.keras.layers.Dense(1, activation='sigmoid') # only 1 node: about the reviews sentiment ( good or bad )
    ])

    ### TRAINING MODEL
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(train_data, train_labels, epochs=3, validation_split=0.2)
    model.save('Recurrent_NN_1.keras')
    prediction = model.evaluate(test_data, test_labels)
    print(prediction)
else:
    model = keras.models.load_model('Recurrent_NN_1.keras')

### PREDICT
# encode user input text to model input format
word_index = datasets.imdb.get_word_index()
#print(len(word_index))
def encode_text(text):

    # text to a list of the text words
    tokens = text_to_word_sequence(text)
    # transform the list of words into list of coresponding integers from word_index or 0
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    # return list of tokens of the same size; [0] is for extracting the first inner list from a nested list; pad_sequences adds 0's
    return sequence.pad_sequences([tokens], MAXLEN)[0]

text = 'that movie was just amazing, so amazing'
encoded = encode_text(text)
#print(encoded)

# in case of a review in integer form, decode it with this: swap keys and values of integer form dictionary
reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]

#print(decode_integers(encoded))

def predict(text):
    encoded_text = encode_text(text)
    # textual data reshape for single row because thats the data format on which model was trained
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_rev = 'I really liked this movie scenes because i had good vibes and the actors were perfect for the story'
#positive_rev = 'That movie was! really loved it and would great watch it again because it was amazingly great'
predict(positive_rev)
negative_rev = 'It wasnt so good especially with the scene that gave me sad feelings and the entire movie was very monotone'
predict(negative_rev)

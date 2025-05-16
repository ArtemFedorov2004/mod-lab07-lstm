import numpy as np
import tensorflow as tf
 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop
 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('input.txt', 'r',encoding='utf-8') as file:
    text = file.read()

text = text.lower().split()

vocabulary = sorted(list(set(text)))


indices_words_dictionary = dict((i, w) for i, w in enumerate(vocabulary))
words_indices_dictionary = dict((w, i) for i, w in enumerate(vocabulary))

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_words.append(text[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, words_indices_dictionary[word]] = 1
    y[i, words_indices_dictionary[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
    start_index = random.randint(0, len(text) - max_length - 1)
    words_list = text[start_index: start_index + max_length]
    generated = words_list.copy()
    for j in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, word in enumerate(words_list):
                x_pred[0, t, words_indices_dictionary[word]] = 1.

            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = indices_words_dictionary[next_index]

            generated.append(next_word)
            words_list.append(next_word)
            words_list.pop(0)

    return ' '.join(generated)

model.fit(X, y, batch_size = 128, epochs = 50)

text_generated_by_lstm = generate_text(1500, 0.2)
print(text_generated_by_lstm)

file_generated_text_write_to = '../result/gen.txt'
with open(file_generated_text_write_to, 'w', encoding='utf-8') as _file_:
    _file_.write(text_generated_by_lstm)
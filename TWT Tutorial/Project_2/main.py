import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb

(train_data, train_label), (test_data, test_label) = data.load_data()

_word_index = keras.datasets.imdb.get_word_index()

word_index = {k:(v+3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#
# # define model
# model = keras.Sequential(
#     [
#         keras.layers.Embedding(880000, 16),
#         keras.layers.GlobalAveragePooling1D(),
#         keras.layers.Dense(16, activation="relu"),
#         keras.layers.Dense(1, activation="sigmoid"),
#     ]
# )
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# x_val = train_data[:10000]
# x_train = train_data[10000:]
#
# y_val = train_label[:10000]
# y_train = train_label[10000:]
#
# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#
# results = model.evaluate(test_data, test_label)
# print(results)
#
# model.save("model.h5")

def review_encode(s):
    encode = [1]
    for word in s:
        if word.lower() in word_index:
            encode.append(word_index[word.lower()])
        else:
            encode.append(2)
    return encode

model = keras.models.load_model("model.h5")

with open('review.txt', encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(',', "").replace('.', "").replace('(', "").replace(')', "").replace(':', "")\
            .replace('\"', "").strip().split()
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])







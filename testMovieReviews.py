import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

#prints the numbers that are associated with the words in the data
#print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

#time to normalize reviews so they all have 200 words

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

def decode_review(text):
        return " ".join([reverse_word_index.get(i,"?") for i in text])

##print(decode_review(test_data[0]))

##print(len(test_data[0]), len(test_data[1]))


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)

    return encoded


model = keras.models.load_model("model.h5")

with open("test.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PAD>"], padding = "post", maxlen = 250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
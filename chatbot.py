import nltk
import numpy
import tensorflow as tf
import keras
import json


with open("intents.json") as f:
    data = json.load(f)

words = []
labels = []
x_all = []
labels_all = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_add = nltk.wordpunct_tokenize(pattern)
        words.extend(words_add)
        x_all.append(words_add)
        labels_all.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])


words = [word.lower() for word in words]

ps = nltk.stem.PorterStemmer()
words = [ps.stem(word) for word in words]

words = sorted(list(set(words)))
labels = sorted(labels)

train_x = []
train_y = []

out_empty = [0] * len(labels)

for x, sentence in enumerate(x_all):
    bag = []
    stem_sent = [ps.stem(w.lower()) for w in sentence]
    for w in words:
        if w in stem_sent:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]

    output_row[labels.index(labels_all[x])] = 1

    train_x.append(bag)
    train_y.append(output_row)

train_x = numpy.asarray(train_x)
train_y = numpy.asarray(train_y)


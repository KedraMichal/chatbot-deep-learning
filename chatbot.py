import nltk
import numpy
import tensorflow as tf
import keras
import json

numpy.set_printoptions(suppress=True)
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

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_shape=[len(words)]))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dense(len(labels), activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=1000, batch_size=8)


def clean_up_sentences(sent):
    sent = nltk.wordpunct_tokenize(sent)
    sent = [ps.stem(word.lower()) for word in sent]

    return sent


def sentence_coding(sent, bag_of_words):
    coded_sentence = []
    for word in bag_of_words:
        if word in sent:
            coded_sentence.append(1)
        else:
            coded_sentence.append(0)

    return coded_sentence



print("Start to talk:")

for i in range(10):
    p = input()
    x = clean_up_sentences(p)
    d = sentence_coding(x, words)
    d = numpy.asarray(d)
    d = d.reshape(1, 50)

    w = int(numpy.argmax(model.predict(d), axis=1))

    for i,p in enumerate(data["intents"]):
        if i == w:
            print(p["responses"][0])

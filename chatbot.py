import nltk
import numpy as np
import tensorflow as tf
import keras
import json
from keras.models import load_model

np.set_printoptions(suppress=True)
with open("data.json") as file:
    data = json.load(file)

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
words = list(set(words))

train_x = []
train_y = []

out_empty = [0] * len(labels)

for i, sentence in enumerate(x_all):
    bag = []
    stem_sent = [ps.stem(w.lower()) for w in sentence]
    for w in words:
        if w in stem_sent:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(labels_all[i])] = 1

    train_x.append(bag)
    train_y.append(output_row)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

try:
    model = load_model("model.h5")
except:
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=[len(words)]))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(len(labels), activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=2000, batch_size=8)

    model.save("model.h5")


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

    return np.asarray(coded_sentence)


def chat():
    print("Start to talk:")
    for i in range(100):
        human_sentence = input()
        human_sentence = clean_up_sentences(human_sentence)
        d = sentence_coding(human_sentence, words)
        d = d.reshape(1, len(words))
        best_answer = int(np.argmax(model.predict(d), axis=1))

        for i, intent in enumerate(data["intents"]):
            if i == best_answer:
                answers = intent["responses"]
                random_answer = np.random.choice(answers)
                print(random_answer)


if __name__ == "__main__":
    chat()


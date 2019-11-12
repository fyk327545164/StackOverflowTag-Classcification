from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Models import *
from configuration import Configuration
import tensorflow as tf


class Vocab:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

        self.nextID = 1

    def getID(self, word):
        return self.word2id[word]

    def getWord(self, id):
        return self.id2word[id]

    def hasWord(self, word):
        return word in self.word2id.keys()


def preprocess(filename, vocab):

    spec = '`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words_list = []
            for s in spec:
                line = line.replace(s, ' ')
            words = line.split()

            for word in words:
                if not vocab.hasWord(word):
                    vocab.word2id[word] = vocab.nextID
                    vocab.id2word[vocab.nextID] = word
                    vocab.nextID += 1
                words_list.append(vocab.word2id[word])
            res.append(words_list)
    return res


class DataLoader:

    def __init__(self, X, Y, padding_idx, seq_length):

        """
        Build batch data loader for training, developing, and testing set
        """

        self.seq_length = seq_length
        self.padding_idx = padding_idx

        self.X = []
        self.Y = []

        self.build(X, Y)

    def build(self, X, Y):

        for x, y in zip(X, Y):
            d = [self.padding_idx for _ in range(self.seq_length)]

            if len(x) <= self.seq_length:
                d[:len(x)] = x[:]
            else:
                d[:] = x[:len(x)]

            label = [0 for _ in range(3)]

            label[y] = 1

            self.X.append(d)
            self.Y.append(label)


def main():

    vocab = Vocab()

    X_aws = preprocess('aws_title.txt', vocab)
    Y_aws = [0 for _ in range(len(X_aws))]
    X_azure = preprocess('azure_title.txt', vocab)
    Y_azure = [1 for _ in range(len(X_azure))]
    X_gcp = preprocess('gcp_title.txt', vocab)
    Y_gcp = [2 for _ in range(len(X_gcp))]

    X = X_aws + X_azure + X_gcp
    Y = Y_aws + Y_azure + Y_gcp

    X, Y = shuffle(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    train = DataLoader(X_train, Y_train, 0, 50)
    test = DataLoader(X_test, Y_test, 0, 50)

    vocab_size = vocab.nextID

    print("Start Training!")

    model = FastText(Configuration('FastText'), vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train.X, train.Y, epochs=2, batch_size=64,
              validation_data=(test.X, test.Y))
    model.summary()

    model = TextCNN(Configuration('TextCNN'), vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train.X, train.Y, epochs=2, batch_size=64,
              validation_data=(test.X, test.Y))
    model.summary()

    model = TextRNN(Configuration('TextRNN'), vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train.X, train.Y, epochs=2, batch_size=64,
              validation_data=(test.X, test.Y))
    model.summary()

    model = TextRCNN(Configuration('TextRCNN'), vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train.X, train.Y, epochs=2, batch_size=64,
              validation_data=(test.X, test.Y))
    model.summary()


main()




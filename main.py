from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Models import *
from configuration import Configuration
import tensorflow as tf
import operator
import numpy as np


class Vocab:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

        self.word_count = {}

    def getID(self, word):
        return self.word2id[word]

    def getWord(self, id):
        return self.id2word[id]

    def hasWord(self, word):
        return word in self.word2id.keys()

    def build(self, filenames):

        spec = '`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    for s in spec:
                        line = line.replace(s, ' ')
                    words = line.split()

                    for word in words:
                        word = word.lower()
                        if word not in self.word_count:
                            self.word_count[word] = 1
                        else:
                            self.word_count[word] += 1

        sorted_dic = sorted(self.word_count.items(), key=operator.itemgetter(1), reverse=True)[:7999]

        self.word2id['<unknown>'] = 1
        self.id2word[1] = '<unknown>'

        index = 2
        for key, _ in sorted_dic:
            self.word2id[key] = index
            self.id2word[index] = key
            index += 1


def preprocess(filename, vocab):

    spec = '`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words_list = []
            # word_s = ""
            for s in spec:
                line = line.replace(s, ' ')
            words = line.split()

            for word in words:
                word = word.lower()
                if not vocab.hasWord(word):
                    word = '<unknown>'
                # word_s = word_s + ' ' + word
                words_list.append(vocab.word2id[word])
            res.append(words_list)
            # res.append(word_s)
    return res


class DataLoader:

    def __init__(self, X, Y, padding_idx, seq_length):

        """
        Build batch data loader for training, developing, and testing set
        """

        self.seq_length = seq_length
        self.padding_idx = padding_idx

        self.X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.seq_length, padding='post')
        self.Y = np.array(Y)


def main():

    vocab = Vocab()

    vocab.build(['aws_title.txt', 'azure_title.txt', 'gcp_title.txt'])

    X_aws = preprocess('aws_title.txt', vocab)
    Y_aws = [[1, 0, 0] for _ in range(len(X_aws))]
    X_azure = preprocess('azure_title.txt', vocab)
    Y_azure = [[0, 1, 0] for _ in range(len(X_azure))]
    X_gcp = preprocess('gcp_title.txt', vocab)
    Y_gcp = [[0, 0, 1] for _ in range(len(X_gcp))]

    X = X_aws + X_azure + X_gcp
    Y = Y_aws + Y_azure + Y_gcp

    X, Y = shuffle(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    train = DataLoader(X_train, Y_train, 0, 50)
    test = DataLoader(X_test, Y_test, 0, 50)

    vocab_size = 8000

    print("Start Training!")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model/fasttext/cp.ckpt',
        verbose=1,
        save_weights_only=True,
        period=1)

    model = SelfAttention(vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()])
    model.fit(train.X, train.Y, epochs=10, batch_size=64,
              validation_data=(test.X, test.Y))
    model.summary()

    # model = FastText(Configuration('FastText'), vocab_size)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()])
    # model.fit(train.X, train.Y, epochs=10, batch_size=64,
    #           validation_data=(test.X, test.Y),
    #           callbacks=[cp_callback])
    # model.summary()
    # model.load_weights('model/fasttext/cp.ckpt')
    # model.evaluate(test.X, test.Y)
    #
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='model/textcnn/cp.ckpt',
    #     verbose=1,
    #     save_weights_only=True,
    #     period=1)
    #
    # model = TextCNN(Configuration('TextCNN'), vocab_size)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()])
    # # model.fit(train.X, train.Y, epochs=10, batch_size=64,
    # #           validation_data=(test.X, test.Y),
    # #           callbacks=[cp_callback])
    # model.load_weights('model/textcnn/cp.ckpt')
    # model.evaluate(test.X, test.Y)
    #
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='model/textrnn/cp.ckpt',
    #     verbose=1,
    #     save_weights_only=True,
    #     period=1)
    #
    # model = TextRNN(Configuration('TextRNN'), vocab_size)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()])
    # # model.fit(train.X, train.Y, epochs=10, batch_size=64,
    # #           validation_data=(test.X, test.Y),
    # #           callbacks=[cp_callback])
    # model.load_weights('model/textrnn/cp.ckpt')
    # model.evaluate(test.X, test.Y)
    #
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='model/textrcnn/cp.ckpt',
    #     verbose=1,
    #     save_weights_only=True,
    #     period=1)
    #
    # model = TextRCNN(Configuration('TextRCNN'), vocab_size)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()])
    # # model.fit(train.X, train.Y, epochs=10, batch_size=64,
    # #           validation_data=(test.X, test.Y),
    # #           callbacks=[cp_callback])
    # model.load_weights('model/textrcnn/cp.ckpt')
    # model.evaluate(test.X, test.Y)


main()




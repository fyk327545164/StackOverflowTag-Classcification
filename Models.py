from tensorflow.keras import Sequential
import tensorflow as tf


class FastText(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(FastText, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.dense1 = tf.keras.layers.Dense(self.config.HIDDEN_DIM, input_shape=(self.config.EMBEDDING_DIM,))

        self.dense2 = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(self.config.HIDDEN_DIM,))

        self.softmax = tf.keras.activations.softmax()

    def call(self, inputs, training=None, mask=None):
        embed = tf.reduce_mean(self.embedding(inputs), 1)

        z = self.dense2(self.dense1(embed))

        return self.softmax(z)


class TextCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.conv1 = Sequential(
            tf.keras.layers.Conv1D(self.config.NUM_CHANNELS * self.config.EMBEDDING_DIM, self.config.kernal_size[0]),
            tf.keras.activations.relu(),
            tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[0] + 1)
        )

        self.conv2 = Sequential(
            tf.keras.layers.Conv1D(self.config.NUM_CHANNELS * self.config.EMBEDDING_DIM, self.config.kernal_size[1]),
            tf.keras.activations.relu(),
            tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[1] + 1)
        )

        self.conv3 = Sequential(
            tf.keras.layers.Conv1D(self.config.NUM_CHANNELS * self.config.EMBEDDING_DIM, self.config.kernal_size[2]),
            tf.keras.activations.relu(),
            tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[2] + 1)
        )

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(self.config.NUM_CHANNELS * 3,))

        self.softmax = tf.keras.activations.softmax()

    def call(self, inputs, training=None, mask=None):
        embed = tf.transpose(self.embedding(inputs), perm=[0, 2, 1])

        conv1 = tf.squeeze(self.conv1(embed), 2)
        conv2 = tf.squeeze(self.conv2(embed), 2)
        conv3 = tf.squeeze(self.conv3(embed), 2)

        conv_seq = tf.concat([conv1, conv2, conv3], axis=-1)
        if training:
            conv_seq = self.dropout(conv_seq)

        out = self.softmax(conv_seq)

        return self.softmax(out)


class TextRNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)


class TextRCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)


class TextRNNAttention(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRNNAttention, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

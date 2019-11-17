import tensorflow as tf


class FastText(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(FastText, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.dense1 = tf.keras.layers.Dense(self.config.HIDDEN_DIM, input_shape=(None, self.config.EMBEDDING_DIM))

        self.dense2 = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.HIDDEN_DIM))

    def call(self, inputs, training=None, mask=None):
        embed = tf.reduce_mean(self.embedding(inputs), 1)

        z = self.dense2(self.dense1(embed))

        return tf.keras.activations.softmax(z)


class TextCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.conv1 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[0], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))
        self.maxPool1 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[0] + 1)

        self.conv2 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[1], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))

        self.maxPool2 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[1] + 1)

        self.conv3 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[2], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))

        self.maxPool3 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[2] + 1)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.NUM_CHANNELS * 3))

    def call(self, inputs, training=None, mask=None):

        embed = self.embedding(inputs)
        conv1 = tf.squeeze(self.maxPool1(self.conv1(embed)), 1)
        conv2 = tf.squeeze(self.maxPool2(self.conv2(embed)), 1)
        conv3 = tf.squeeze(self.maxPool3(self.conv3(embed)), 1)

        conv_seq = tf.concat([conv1, conv2, conv3], axis=-1)

        if training:
            conv_seq = self.dropout(conv_seq)

        out = self.dense(conv_seq)

        return tf.keras.activations.softmax(out)


class TextRNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.forward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM)
        self.backward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, go_backwards=True)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.RNN_DIM*2))

    def call(self, inputs, training=None, mask=None):

        embed = self.embedding(inputs)

        forward_out = self.forward_LSTM(embed)
        backward_out = self.backward_LSTM(embed)

        out = tf.concat([forward_out, backward_out], axis=-1)

        if training:
            out = self.dropout(out)

        out = self.dense(out)

        return tf.keras.activations.softmax(out)


class TextRCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.forward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, return_sequences=True)
        self.backward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, return_sequences=True, go_backwards=True)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense1 = tf.keras.layers.Dense(self.config.HIDDEN_DIM, input_shape=(None, self.config.RNN_DIM * 2 + self.config.EMBEDDING_DIM),
                                            activation='tanh')

        self.dense2 = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.HIDDEN_DIM))

    def call(self, inputs, training=None, mask=None):

        embed = self.embedding(inputs)

        forward_out = self.forward_LSTM(embed)
        backward_out = self.backward_LSTM(embed)

        out = tf.concat([forward_out, embed, backward_out], axis=-1)

        out = self.dense1(out)

        out = tf.keras.layers.MaxPool1D(self.config.MAX_LENGTH)(out)

        if training:
            out = self.dropout(out)

        out = self.dense2(tf.squeeze(out, 1))

        return tf.keras.activations.softmax(out)

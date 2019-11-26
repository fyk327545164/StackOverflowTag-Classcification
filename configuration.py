

class Configuration:
    def __init__(self, mode):

        if mode == 'FastText':

            self.EMBEDDING_DIM = 128
            self.HIDDEN_DIM = 64
            self.OUTPUT_DIM = 3

        elif mode == 'TextCNN':

            self.EMBEDDING_DIM = 128
            self.kernal_size = [3, 4, 5]
            self.NUM_CHANNELS = 100
            self.MAX_LENGTH = 20
            self.OUTPUT_DIM = 3

        elif mode == 'TextRNN':

            self.EMBEDDING_DIM = 128
            self.RNN_DIM = 128
            self.OUTPUT_DIM = 3

        elif mode == 'TextRCNN':

            self.EMBEDDING_DIM = 128
            self.RNN_DIM = 128
            self.HIDDEN_DIM = 64
            self.MAX_LENGTH = 20
            self.OUTPUT_DIM = 3

        elif mode == 'SelfAttention':

            self.EMBEDDING_DIM = 128

            self.num_layer = 3
            self.num_heads = 4

            self.dim = 32

            self.OUTPUT_DIM = 3

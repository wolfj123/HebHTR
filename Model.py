import tensorflow as tf
from typing import Tuple
import numpy as np
from word_beam_search import WordBeamSearch

class DecoderType:
    BestPath = 0
    WordBeamSearch = 1

class Model:
    batchSize: int
    imgSize: Tuple[int, int]
    maxTextLen: int

    def __init__(self, charList: list[str], batchSize: int, imgSize: Tuple[int, int], maxTextLen: int, decoderType=DecoderType.BestPath, mustRestore: bool = False) -> None:
        self.charList = charList
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.maxTextLen = maxTextLen
        self.decoderType = decoderType
        self.mustRestore = mustRestore  # Store the mustRestore flag

        # Input layers
        self.inputImgs = tf.keras.Input(shape=(self.imgSize[0], self.imgSize[1]), name='input_imgs')

        # CNN
        cnn_output = self._build_cnn(self.inputImgs)

        # RNN
        rnn_output = self._build_rnn(cnn_output)

        # Output layer (logits for CTC)
        logits = tf.keras.layers.Dense(units=len(self.charList) + 1)(rnn_output)
        self.output = tf.keras.layers.Activation('softmax')(logits)

        # Model
        self.model = tf.keras.Model(inputs=self.inputImgs, outputs=self.output)

        # Restore weights if mustRestore is True
        if self.mustRestore:
            self._restore_weights()

    def _restore_weights(self):
        """Restore model weights from a checkpoint."""
        checkpoint_path = './model_checkpoint.h5'  # Example checkpoint path
        try:
            self.model.load_weights(checkpoint_path)
            print(f"Model weights restored from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to restore model weights: {e}")

    def _build_cnn(self, inputImgs):
        # Use a Lambda layer to expand dimensions
        input_4d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(inputImgs)

        def conv_layer(x, filters, kernel_size, strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2)):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=None)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            if pool_size:
                x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides)(x)
            return x

        x = conv_layer(input_4d, 64, (3, 3))
        x = conv_layer(x, 128, (3, 3))
        x = conv_layer(x, 256, (3, 3))
        x = conv_layer(x, 256, (3, 3), pool_size=(2, 1), pool_strides=(2, 1))
        x = conv_layer(x, 512, (3, 3))
        x = conv_layer(x, 512, (3, 3), pool_size=(2, 1), pool_strides=(2, 1))
        x = conv_layer(x, 512, (2, 2), pool_size=None)

        cnn_output = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
        return cnn_output

    def _build_rnn(self, cnn_output):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, return_sequences=True)
        )(cnn_output)
        return x

    def decode(self, y_pred, input_lengths):
        if self.decoderType == DecoderType.BestPath:
            decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_lengths, greedy=True)
            return decoded[0]

        elif self.decoderType == DecoderType.WordBeamSearch:
            word_beam_search_module = tf.load_op_library('./TFWordBeamSearch.so')
            softmax_out = y_pred.numpy()
            chars = ''.join(self.charList)
            with open('model/wordCharList.txt', 'r', encoding='utf-8') as f:
                wordChars = f.read().splitlines()[0]
            corpus = open('data/corpus.txt', 'r', encoding='utf-8').read()
            decoded = word_beam_search_module.word_beam_search(
                softmax_out,
                beamWidth=50,
                mode='Words',
                corpus=corpus.encode('utf-8'),
                chars=chars.encode('utf-8'),
                wordChars=wordChars.encode('utf-8')
            )
            return decoded

    def summary(self):
        self.model.summary()
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
        """Restore model weights from a TensorFlow checkpoint."""
        checkpoint_path = './model/snapshot-6'  # Path to the checkpoint (without extensions)
        try:
            # Use TensorFlow's Checkpoint to load weights
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(checkpoint_path).expect_partial()
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
            # Perform CTC decoding
            decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_lengths, greedy=True)

            # Check if the output is a SparseTensor
            if isinstance(decoded[0], tf.SparseTensor):
                decoded_dense = tf.sparse.to_dense(decoded[0]).numpy()
            else:
                decoded_dense = decoded[0]  # Already a NumPy array

            # Map indices to characters
            result = []
            for sequence in decoded_dense:
                text = ''.join([self.charList[idx] for idx in sequence if idx != -1])  # Ignore padding (-1)
                result.append(text)
            return result

        elif self.decoderType == DecoderType.WordBeamSearch:
            # Word Beam Search decoding
            chars = ''.join(self.charList)  # Characters that can be recognized
            with open('model/wordCharList_utf8.txt', 'r', encoding='utf-8') as f:
                wordChars = f.read().strip()  # Characters that form words
            with open('data/corpus.txt', 'r', encoding='utf-8') as f:
                corpus = f.read().strip()  # Corpus for language model

            print("y_pred shape:", y_pred.shape)  # Should be (batch_size, time_steps, num_classes)
            print("Number of characters in chars:", len(chars))

            # Initialize WordBeamSearch
            wbs = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

            # Compute label string using WordBeamSearch
            label_str = wbs.compute(y_pred)  # y_pred is already a NumPy array

            # Map indices to characters
            result = []
            for sequence in label_str:
                text = ''.join([self.charList[idx] for idx in sequence if idx != -1])  # Ignore padding (-1)
                result.append(text)
            return result
        
    def summary(self):
        self.model.summary()
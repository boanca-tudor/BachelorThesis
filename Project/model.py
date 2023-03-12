import tensorflow as tf
from tensorflow.python import keras
import numpy as np
from tensorflow.python.keras.layers import *


class Encoder(Layer):
    def __init__(self, z_channels):
        super(Encoder, self).__init__()
        self.encoder_layers = keras.Sequential()
        self.create_convolutions()
        self.create_dense(z_channels)

        self.encoder_layers.build((1, 128, 128, 3))
        self.encoder_layers.summary()

    def __call__(self, x):
        return self.encoder_layers(x)

    def create_convolutions(self):
        self.encoder_layers.add(InputLayer(input_shape=(128, 128, 3)))
        self.encoder_layers.add(Conv2D(64, kernel_size=5, strides=2, activation='relu', padding='same',
                                       name='e_conv1'))
        self.encoder_layers.add(Conv2D(128, kernel_size=5, strides=2, activation='relu', padding='same',
                                       name='e_conv2'))
        self.encoder_layers.add(Conv2D(256, kernel_size=5, strides=2, activation='relu', padding='same',
                                       name='e_conv3'))
        self.encoder_layers.add(Conv2D(512, kernel_size=5, strides=2, activation='relu', padding='same',
                                       name='e_conv4'))
        self.encoder_layers.add(Conv2D(1024, kernel_size=5, strides=2, activation='relu', padding='same',
                                       name='e_conv5'))

    def create_dense(self, z_channels):
        self.encoder_layers.add(Flatten())
        self.encoder_layers.add(Dense(z_channels, activation='tanh', name='e_fc1'))

class Decoder(Layer):
    def __init__(self, zl_channels, gen_channels):
        super(Decoder, self).__init__()
        self.decoder_layers = keras.Sequential()
        self.create_dense(gen_channels, zl_channels)
        self.create_deconvolutions()

        self.decoder_layers.build((1, zl_channels))
        self.decoder_layers.summary()

    def __call__(self, x):
        return self.decoder_layers(x)

    def create_dense(self, gen_channels, zl_channels):
        self.decoder_layers.add(InputLayer(input_shape=(1, zl_channels)))
        self.decoder_layers.add(Dense(gen_channels * 4 ** 2, activation='relu'))
        self.decoder_layers.add(Reshape((4, 4, gen_channels)))

    def create_deconvolutions(self):
        self.decoder_layers.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.decoder_layers.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.decoder_layers.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.decoder_layers.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.decoder_layers.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='relu'))


class Autoencoder(keras.Model):
    def __init__(self, z_channels, l_channels, gen_channels):
        super().__init__()
        self.encoder = Encoder(z_channels)
        self.decoder = Decoder(z_channels + l_channels, gen_channels)

    def __call__(self, x, age):
        x = self.encoder(x)
        age_label = -np.ones((1, 10))
        age_index = age // 10
        age_label[0, age_index] = 1
        x = tf.concat([x, age_label], 1)
        x = self.decoder(x)
        return x

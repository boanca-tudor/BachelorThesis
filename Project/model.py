import tensorflow as tf
from tensorflow.python import keras
import numpy as np
from tensorflow.python.keras.layers import *


class Encoder(Layer):
    def __init__(self, z_channels):
        super(Encoder, self).__init__()
        self.encoder_layers = keras.Sequential(name="e_layers")
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
        self.encoder_layers.add(Flatten(name='e_flatten'))
        self.encoder_layers.add(Dense(z_channels, activation='tanh', name='e_fc1'))


class Decoder(Layer):
    def __init__(self, zl_channels, gen_channels):
        super(Decoder, self).__init__()
        self.decoder_layers = keras.Sequential(name="g_layers")
        self.create_dense(gen_channels, zl_channels)
        self.create_deconvolutions()

        self.decoder_layers.build((1, zl_channels))
        self.decoder_layers.summary()

    def __call__(self, x):
        return self.decoder_layers(x)

    def create_dense(self, gen_channels, zl_channels):
        self.decoder_layers.add(InputLayer(input_shape=(1, zl_channels)))
        self.decoder_layers.add(Dense(gen_channels * 4 ** 2, activation='relu', name='g_fc1'))
        self.decoder_layers.add(Reshape((4, 4, gen_channels)))

    def create_deconvolutions(self):
        self.decoder_layers.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', activation='relu',
                                                name='g_conv1'))
        self.decoder_layers.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu',
                                                name='g_conv2'))
        self.decoder_layers.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', activation='relu',
                                                name='g_conv3'))
        self.decoder_layers.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='relu',
                                                name='g_conv4'))
        self.decoder_layers.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', activation='relu',
                                                name='g_conv5'))
        self.decoder_layers.add(Conv2DTranspose(16, kernel_size=5, strides=1, padding='same', activation='relu',
                                                name='g_conv6'))
        self.decoder_layers.add(Conv2DTranspose(3, kernel_size=1, strides=1, padding='same', activation='tanh',
                                                name='g_conv7'))


class DiscriminatorZ(Layer):
    def __init__(self, z_channels):
        super(DiscriminatorZ, self).__init__()
        self.discriminator_layers = keras.Sequential(name="dz_layers")
        self.create_dense_layers(z_channels)

        self.discriminator_layers.build((1, z_channels))
        self.discriminator_layers.summary()

    def create_dense_layers(self, z_channels):
        self.discriminator_layers.add(InputLayer(input_shape=(z_channels, )))
        self.discriminator_layers.add(Dense(64, activation='relu', name='dz_fc1'))
        self.discriminator_layers.add(Dense(32, activation='relu', name='dz_fc2'))
        self.discriminator_layers.add(Dense(16, activation='relu', name='dz_fc3'))
        self.discriminator_layers.add(Dense(1, activation='sigmoid', name='dz_fc4'))

    def __call__(self, x):
        return self.discriminator_layers(x)


class DiscriminatorImg(Layer):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        self.pre_concat_discriminator_layers = keras.Sequential(name='dimg_preconcat_layers')
        self.post_concat_discriminator_layers = keras.Sequential(name='dimg_postconcat_layers')

        self.create_convolutional_layers()
        self.create_dense_layers()

        self.pre_concat_discriminator_layers.build((1, 128, 128, 3))
        self.pre_concat_discriminator_layers.summary()

        self.post_concat_discriminator_layers.build((1, 64, 64, 26))
        self.post_concat_discriminator_layers.summary()

    def create_convolutional_layers(self):
        self.pre_concat_discriminator_layers.add(InputLayer(input_shape=(128, 128, 3)))
        self.pre_concat_discriminator_layers.add(Conv2D(16, kernel_size=2, strides=2, padding='same',
                                                        activation='relu', name='dimg_conv1'))
        # TODO replace hardcoded with age_count
        self.post_concat_discriminator_layers.add(InputLayer(input_shape=(64, 64, 16 + 10)))
        self.post_concat_discriminator_layers.add(Conv2D(32, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv2'))
        self.post_concat_discriminator_layers.add(Conv2D(64, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv3'))
        self.post_concat_discriminator_layers.add(Conv2D(128, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv4'))

    def create_dense_layers(self):
        self.post_concat_discriminator_layers.add(Flatten())
        self.post_concat_discriminator_layers.add(Dense(1024, activation='relu', name='dimg_fc1'))
        self.post_concat_discriminator_layers.add(Dense(1, activation='relu', name='dimg_fc2'))

    def __call__(self, x, age):
        x = self.pre_concat_discriminator_layers(x)
        reshaped_age_label = -np.ones(shape=(64, 64, 10))
        reshaped_age_label[:, :, age // 10] = 1
        reshaped_age_label = tf.cast(tf.expand_dims(reshaped_age_label, axis=0), dtype=tf.float32)
        x = tf.concat([x, reshaped_age_label], -1)
        return self.post_concat_discriminator_layers(x)


class CAAE(keras.Model):
    def __init__(self, z_channels, l_channels, gen_channels):
        super().__init__()
        self.encoder = Encoder(z_channels)
        self.discriminatorZ = DiscriminatorZ(z_channels)
        self.decoder = Decoder(z_channels + l_channels, gen_channels)
        self.discriminatorImg = DiscriminatorImg()

    def __call__(self, x, age):
        x = self.encoder(x)
        dz = self.discriminatorZ(x)
        age_label = -np.ones((1, 10))
        age_index = age // 10
        age_label[0, age_index] = 1
        x = tf.concat([x, age_label], 1)
        x = self.decoder(x)
        dimg = self.discriminatorImg(x, age)
        return x, dz, dimg

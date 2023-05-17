import datetime

import keras
from keras.metrics import *
from keras.layers import *
from keras.losses import *
from keras.optimizers.schedules.learning_rate_schedule import *
from keras.optimizers.optimizer_v2.adam import *
from model_utils import *
import time
from skimage.io import imsave
import matplotlib.pyplot as plt


class Encoder(keras.Model):
    def __init__(self, z_channels):
        super(Encoder, self).__init__()
        self.encoder_layers = keras.Sequential(name="e_layers")
        self.create_convolutions()
        self.create_dense(z_channels)

        self.encoder_layers.build((1, 128, 128, 3))
        # self.encoder_layers.summary()

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


class Decoder(keras.Model):
    def __init__(self, zl_channels, gen_channels):
        super(Decoder, self).__init__()
        self.decoder_layers = keras.Sequential(name="g_layers")
        self.create_dense(gen_channels, zl_channels)
        self.create_deconvolutions()

        self.decoder_layers.build((1, zl_channels))
        # self.decoder_layers.summary()

    def __call__(self, x):
        return self.decoder_layers(x)

    def create_dense(self, gen_channels, zl_channels):
        self.decoder_layers.add(InputLayer(input_shape=zl_channels))
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


class DiscriminatorZ(keras.Model):
    def __init__(self, z_channels):
        super(DiscriminatorZ, self).__init__()
        self.discriminator_layers = keras.Sequential(name="dz_layers")
        self.create_dense_layers(z_channels)

        self.discriminator_layers.build((1, z_channels))
        # self.discriminator_layers.summary()

    def create_dense_layers(self, z_channels):
        self.discriminator_layers.add(InputLayer(input_shape=(z_channels, )))
        self.discriminator_layers.add(Dense(64, activation='relu', name='dz_fc1'))
        self.discriminator_layers.add(BatchNormalization(name='dz_bn1'))
        self.discriminator_layers.add(Dense(32, activation='relu', name='dz_fc2'))
        self.discriminator_layers.add(BatchNormalization(name='dz_bn2'))
        self.discriminator_layers.add(Dense(16, activation='relu', name='dz_fc3'))
        self.discriminator_layers.add(BatchNormalization(name='dz_bn3'))
        self.discriminator_layers.add(Dense(1, name='dz_fc4'))

    def __call__(self, x):
        x = self.discriminator_layers(x)
        logit_layer = Activation('sigmoid')
        return logit_layer(x)


class DiscriminatorImg(keras.Model):
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
        self.pre_concat_discriminator_layers.add(BatchNormalization(name='dimg_bn1'))
        # TODO replace hardcoded with age_count
        self.post_concat_discriminator_layers.add(InputLayer(input_shape=(64, 64, 16 + 10)))
        self.post_concat_discriminator_layers.add(Conv2D(32, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv2'))
        self.pre_concat_discriminator_layers.add(BatchNormalization(name='dimg_bn2'))
        self.post_concat_discriminator_layers.add(Conv2D(64, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv3'))
        self.pre_concat_discriminator_layers.add(BatchNormalization(name='dimg_bn3'))
        self.post_concat_discriminator_layers.add(Conv2D(128, kernel_size=2, strides=2, padding='same',
                                                         activation='relu', name='dimg_conv4'))
        self.pre_concat_discriminator_layers.add(BatchNormalization(name='dimg_bn4'))

    def create_dense_layers(self):
        self.post_concat_discriminator_layers.add(Flatten())
        self.post_concat_discriminator_layers.add(Dense(1024, activation='relu', name='dimg_fc1'))
        self.post_concat_discriminator_layers.add(Dense(1, name='dimg_fc2'))

    def __call__(self, x, ages):
        x = self.pre_concat_discriminator_layers(x)
        reshaped_age_label = reshape_age_tensor_to_4d(ages, 64, 64)
        reshaped_age_label = tf.cast(reshaped_age_label, dtype=tf.float32)
        x = tf.concat([x, reshaped_age_label], -1)
        x = self.post_concat_discriminator_layers(x)
        logit_layer = Activation('sigmoid')
        return logit_layer(x)


class PatchDiscriminator(keras.Model):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.pre_concat_discriminator_layers = keras.Sequential(name='patchd_preconcat_layers')
        self.post_concat_discriminator_layers = keras.Sequential(name='patchd_postconcat_layers')

        self.create_convolutional_layers()

        self.pre_concat_discriminator_layers.build((1, 128, 128, 3))
        self.pre_concat_discriminator_layers.summary()

        self.post_concat_discriminator_layers.build((1, 64, 64, 74))
        self.post_concat_discriminator_layers.summary()


    def create_convolutional_layers(self):
        self.pre_concat_discriminator_layers.add(InputLayer(input_shape=(128, 128, 3)))
        self.pre_concat_discriminator_layers.add(ZeroPadding2D(padding=(1, 1), name="patchd_preconcat_padding1"))
        self.pre_concat_discriminator_layers.add(Conv2D(64, kernel_size=4, strides=2, padding="valid",
                                                        activation='relu', name='patchd_preconcat_conv1'))
        self.pre_concat_discriminator_layers.add(BatchNormalization(name='patchd_preconcat_bn1'))

        self.post_concat_discriminator_layers.add(InputLayer(input_shape=(64, 64, 64 + 10)))
        for i in range(1, 4):
            nf_mult = min(2 ** i, 8)
            self.post_concat_discriminator_layers.add(ZeroPadding2D(padding=(1, 1), name=f"patchd_postconcat_padding{i}"))
            self.post_concat_discriminator_layers.add(Conv2D(64 * nf_mult + (10 if i == 1 else 0),
                                                      kernel_size=4, strides=2, use_bias=True,
                                                             name=f"patchd_postconcat_conv{i}"))
            self.post_concat_discriminator_layers.add(LeakyReLU(0.2, name=f"patchd_postconcat_lrelu{i}"))

        self.post_concat_discriminator_layers.add(ZeroPadding2D(padding=(1, 1), name=f"patchd_postconcat_padding5"))
        self.post_concat_discriminator_layers.add(Conv2D(1, kernel_size=4, strides=1, name=f"patchd_postconcat_conv5"))

    def __call__(self, x, ages):
        x = self.pre_concat_discriminator_layers(x)
        reshaped_age_label = reshape_age_tensor_to_4d(ages, 64, 64)
        reshaped_age_label = tf.cast(reshaped_age_label, dtype=tf.float32)
        x = tf.concat([x, reshaped_age_label], -1)
        x = self.post_concat_discriminator_layers(x)
        sigmoid_layer = Activation('sigmoid')
        return sigmoid_layer(x)


# dataset_size is hardcoded for UTKFace
class CAAE(keras.Model):
    def __init__(self, z_channels, l_channels, gen_channels, dataset_size=23708):
        super().__init__()
        self.z_channels = z_channels
        self.l_channels = l_channels
        self.gen_channels = gen_channels
        self.encoder = Encoder(z_channels)
        self.discriminatorZ = DiscriminatorZ(z_channels)
        self.decoder = Decoder(z_channels + l_channels, gen_channels)
        # self.discriminatorImg = DiscriminatorImg()
        self.patchDiscriminator = PatchDiscriminator()

        # losses
        self.loss_l2 = MeanSquaredError()
        self.loss_l1 = MeanAbsoluteError()
        self.loss_bce = BinaryCrossentropy()

        # optimizers
        self.learning_rate = ExponentialDecay(initial_learning_rate=0.0002,
                                              decay_steps=dataset_size / 64 * 2,
                                              decay_rate=1.0,
                                              staircase=True)
        self.eg_optimizer = Adam(self.learning_rate, beta_1=0.5)
        self.dz_optimizer = Adam(self.learning_rate, beta_1=0.5)
        self.dimg_optimizer = Adam(self.learning_rate, beta_1=0.5)

        self.eg_tracker = Mean(name='eg_loss')
        self.dz_tracker = Mean(name='dz_loss')
        self.dimg_tracker = Mean(name='dimg_loss')

    def eval(self, args):
        x = args[0]
        age = args[1]
        x = self.encoder(x)
        age_label = tf.expand_dims(create_age_tensor(age), axis=0)
        x = tf.concat([x, age_label], 1)
        x = self.decoder(x)
        return x

    # data is image batch + labels
    @tf.function
    def train_step(self, data, loss_weights=(0, 0.0001, 0.0001)):
        images = data[0]
        labels = data[1]

        with tf.GradientTape() as eg_tape, tf.GradientTape() as dz_tape, tf.GradientTape() as dimg_tape:
            z_images = self.encoder(images)

            zl_images = tf.concat([z_images, labels], 1)
            generated = self.decoder(zl_images)

            # input/output loss
            eg_loss = self.loss_l2(images, generated)
            # total variance loss
            tv_loss = loss_weights[0] * (self.loss_l1(generated[:, :, :, :-1], generated[:, :, :, 1:]) +
                                         self.loss_l1(generated[:, :, :-1, :], generated[:, :, 1:, :]))

            # discriminatorZ loss
            z_prior = np.random.uniform(-1, 1, z_images.shape).astype(np.float32)
            dz_prior_logits = self.discriminatorZ(z_prior)
            dz_logits = self.discriminatorZ(z_images)
            dz_loss_prior = self.loss_bce(dz_prior_logits, tf.ones_like(dz_prior_logits))
            dz_loss = self.loss_bce(dz_logits, tf.zeros_like(dz_logits))
            dz_total_loss = dz_loss + dz_loss_prior

            # discriminatorZ/encoder loss
            ez_loss = loss_weights[1] * self.loss_bce(dz_logits, tf.ones_like(dz_logits))

            # discriminatorImg loss
            dimg_input_logits = self.patchDiscriminator(images, labels)
            dimg_output_logits = self.patchDiscriminator(generated, labels)

            dimg_input_loss = self.loss_bce(dimg_input_logits, tf.ones_like(dimg_input_logits))
            dimg_output_loss = self.loss_bce(dimg_output_logits, tf.zeros_like(dimg_output_logits))
            dimg_total_loss = dimg_input_loss + dimg_output_loss

            # discriminatorImg/generator loss
            dg_loss = loss_weights[2] * self.loss_bce(dimg_output_logits, tf.ones_like(dimg_output_logits))

            eg_total_loss = eg_loss + dg_loss + ez_loss + tv_loss

        eg_gradients = eg_tape.gradient(eg_total_loss,
                                        self.encoder.trainable_variables + self.decoder.trainable_variables)
        dz_gradient = dz_tape.gradient(dz_total_loss,
                                       self.discriminatorZ.trainable_variables)
        dimg_gradient = dimg_tape.gradient(dimg_total_loss,
                                           self.patchDiscriminator.trainable_variables)

        self.eg_optimizer.apply_gradients(zip(eg_gradients,
                                          self.encoder.trainable_variables + self.decoder.trainable_variables))
        self.dz_optimizer.apply_gradients(zip(dz_gradient,
                                          self.discriminatorZ.trainable_variables))
        self.dimg_optimizer.apply_gradients(zip(dimg_gradient,
                                            self.patchDiscriminator.trainable_variables))

        self.eg_tracker.update_state(eg_total_loss)
        self.dz_tracker.update_state(dz_total_loss)
        self.dimg_tracker.update_state(dimg_total_loss)

    def train(self, epochs, dataset_path, batch_size, save="all", previous_path=None):
        image_paths = list_full_paths(dataset_path)

        eg_losses = []
        dz_losses = []
        dimg_losses = []

        day = datetime.date.today()

        checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            discriminator_z=self.discriminatorZ,
            patch_discriminator=self.patchDiscriminator,
            eg_optimizer=self.eg_optimizer,
            dz_optimizer=self.dz_optimizer,
            dimg_optimizer=self.dimg_optimizer,
        )

        previous_epoch_count = None
        if previous_path is not None:
            self.load_model(previous_path)
            previous_epoch_count = int((previous_path.split("/")[1]).split("_")[0])

        for epoch in range(epochs):
            start = time.time()
            np.random.shuffle(image_paths)
            starting_index = 0
            ending_index = batch_size
            while ending_index <= len(image_paths):
                images, ages = read_images(image_paths[starting_index: ending_index])
                self.train_step((tf.convert_to_tensor(images), tf.convert_to_tensor(ages)))
                starting_index = ending_index
                if ending_index != len(image_paths):
                    ending_index = min(ending_index + batch_size, len(image_paths))
                else:
                    ending_index += 1

            if save == "all":
                self.save_results(checkpoint, dataset_path, day, epoch, image_paths, previous_epoch_count)
            elif save.startswith("every"):
                count = int(save[5:])
                if (epoch + 1) % count == 0:
                    self.save_results(checkpoint, dataset_path, day, epoch, image_paths, previous_epoch_count)

            eg_losses.append(self.eg_tracker.result().numpy())
            dz_losses.append(self.dz_tracker.result().numpy())
            dimg_losses.append(self.dimg_tracker.result().numpy())

            print(f"Epoch {epoch + 1} - Elapsed time : {time.time() - start}")

        plt.plot(eg_losses, label='EG Losses')
        plt.plot(dz_losses, label='DZ Losses')
        plt.plot(dimg_losses, label='DIMG Losses')

        plt.legend()

        plt.savefig(f"{day}/{epochs}_epochs_{dataset_path.split('/')[1]}/losses.png")

    def save_results(self, checkpoint, dataset_path, day, epoch, image_paths, previous_epoch_count=None):
        if previous_epoch_count is not None:
            epoch += previous_epoch_count
        folder_string = f"{day}/{epoch + 1}_epochs_{dataset_path.split('/')[1]}/"
        checkpoint.save(folder_string)
        self.__view_progress(image_paths, folder_string)

    def __view_progress(self, data, folder_string):
        test = np.random.choice(data, 16)
        ages = create_all_ages()
        images, _ = read_images(test)
        all_results = []
        for image in images:
            results = [image]
            for age in ages:
                image_tensor = tf.expand_dims(image, axis=0)
                results.append(tf.squeeze(self.eval([image_tensor, age])).numpy())

            all_results.append(np.concatenate(results))

        all_results = np.asarray(all_results)
        final_result = all_results[0]
        for i in range(1, len(all_results)):
            final_result = np.concatenate((final_result, all_results[i]), axis=1)
        imsave(f"{folder_string}/result.jpg", (((final_result + 1) / 2) * 255).astype(np.uint8))

    def load_model(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            discriminatorZ=self.discriminatorZ,
            discriminatorImg=self.discriminatorImg,
            eg_optimizer=self.eg_optimizer,
            dz_optimizer=self.dz_optimizer,
            dimg_optimizer=self.dimg_optimizer,
        )
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

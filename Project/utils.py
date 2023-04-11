import tensorflow as tf
import os


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def read_batch(batch_files_array):
    images = []
    for batch_file in batch_files_array:
        images.append(load_image(tf.convert_to_tensor(batch_file)))
    return tf.convert_to_tensor(images)


@tf.function
def load_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw)
    resize_image = tf.image.resize(image, (128, 128))
    return resize_image / 255

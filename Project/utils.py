import tensorflow as tf
import os
import numpy as np
from PIL import Image
from matplotlib import cm


def array_to_image(array):
    return Image.fromarray(np.uint8(array)).convert('RGB')


def concatenate_multiple_images(image_array):
    dst = Image.new('RGB', (0, 0))
    for next_image in image_array:
        dst = concatenate_images(dst, array_to_image(next_image))
    return dst


def concatenate_images(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def create_all_ages():
    ages = []
    for i in range(10):
        ages.append(i * 10 + 5)
    return np.asarray(ages)


def label_age(age: int):
    return min(age // 10, 9)


def create_age_tensor(age: int):
    age_tensor = -np.ones((10,), dtype='float32')
    age = label_age(age)
    age_tensor[age] *= -1
    return age_tensor


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def read_images(image_filenames):
    images = []
    ages = []
    for image_filename in image_filenames:
        ages.append(create_age_tensor(int(image_filename.split("/")[-1].split("_")[0])))
        images.append(load_image(image_filename))
    return np.asarray(images), np.asarray(ages)


def load_image(filename):
    img = Image.open(filename).resize((128, 128))
    return np.asarray(img).astype(dtype='float32') / 255


@tf.function
def reshape_age_tensor_to_4d(tensor, dim1, dim2):
    unpacked = tf.unstack(tensor)
    results = []
    for next in unpacked:
        tiled = tf.tile(next, [dim2])
        next = tf.reshape(tiled, [dim2, next.shape[0]])
        tiled = tf.tile(next, [1, dim1])
        next = tf.reshape(tiled, [dim1, dim2, next.shape[1]])
        results.append(next)
    return tf.stack(results)

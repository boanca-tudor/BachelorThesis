import time

import tensorflow as tf
from facepplib.exceptions import ResourceAttrError, ResourceSetIndexError
from skimage.io import imsave
import numpy as np

from facepp_utils import *
from models.caae import CAAE
from models.model_utils import *


def load_model(model_path):
    model = CAAE(z_channels=100,
                 l_channels=10,
                 gen_channels=1024)

    model.load_model(model_path)

    return model


def generate_image(model, base_image_url, age, target_filepath):
    image = load_image(base_image_url)
    image = tf.expand_dims(image, axis=0)
    result = tf.squeeze(model.eval([image, age])).numpy()

    imsave(target_filepath, (((result + 1) / 2) * 255).astype(np.uint8))


def for_one_image(facepp_client, image_url, model, age):
    target_filepath = "metric_images/result.jpg"
    generate_image(model, image_url, age, target_filepath)

    cmp = facepp_client.image.get(image_file=target_filepath)

    try:
        return cmp.faces[0].age['value']
    except ResourceAttrError:
        return np.nan
    except ResourceSetIndexError:
        return np.nan


def for_more_images(facepp_client, images, model_path, age):
    model = load_model(model_path)

    ages = []
    for image in images:
        ages.append(for_one_image(facepp_client, image, model, age))
        time.sleep(.5)

    return np.nanmean(ages)


if __name__ == "__main__":
    facepp_client = create_facepp_client('face++_config.ini')
    model_paths = read_caae_paths('face++_config.ini')

    images21_30 = np.load('21-30.npy')
    images31_40 = np.load('31-40.npy')
    images41_50 = np.load('41-50.npy')
    images51 = np.load('51+.npy')

    age = 55
    model_path = '../../' + model_paths['patchwgan']
    images = images41_50
    print(for_more_images(facepp_client, images, model_path, age))


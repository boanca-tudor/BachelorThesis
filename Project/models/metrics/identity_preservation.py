import json

import tensorflow as tf
from skimage.io import imsave
import numpy as np

from facepp_utils import *
from models.caae import CAAE
from models.model_utils import load_image


def load_model(model_path):
    model =  CAAE(z_channels=100,
                  l_channels=10,
                  gen_channels=1024)

    model.load_model(model_path)

    return model


def generate_image(model, base_image_url, age, target_filepath):
    image = load_image(base_image_url)
    image = tf.expand_dims(image, axis=0)
    result = tf.squeeze(model.eval([image, age])).numpy()

    imsave(target_filepath, (((result + 1) / 2) * 255).astype(np.uint8))


def for_one_image(facepp_client, image_url, model_path, age):
    model = load_model(model_path)
    target_filepath = "metric_images/result.jpg"
    generate_image(model, image_url, age, target_filepath)

    cmp = facepp_client.compare.get(image_file1=image_url,
                                    image_file2=target_filepath)

    print('thresholds', '=', json.dumps(cmp.thresholds, indent=4))
    print('confidence', '=', cmp.confidence)

    return cmp.confidence, json.dumps(cmp.thresholds)


if __name__ == "__main__":
    facepp_client = create_facepp_client('face++_config.ini')
    model_paths = read_caae_paths('face++_config.ini')

    for_one_image(facepp_client, 'metric_images/test.jpg', model_paths['base'], 40)

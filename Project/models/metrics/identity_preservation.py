import tensorflow as tf
from facepplib.exceptions import ResourceAttrError
from skimage.io import imsave
import numpy as np

from facepp_utils import *
from models.caae import CAAE
from models.model_utils import *


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


def for_one_image(facepp_client, image_url, model, age):
    target_filepath = "metric_images/result.jpg"
    generate_image(model, image_url, age, target_filepath)

    cmp = facepp_client.compare.get(image_file1=image_url,
                                    image_file2=target_filepath)

    try:
        return cmp.confidence, cmp.thresholds["1e-5"]
    except ResourceAttrError:
        return np.nan, np.nan


def for_more_images(facepp_client, dataset_url, model_path, age, count):
    image_paths = list_full_paths(dataset_url)
    np.random.shuffle(image_paths)
    image_paths = image_paths[:count]

    model = load_model(model_path)

    confidences = []
    thresholds = []
    for image in image_paths:
        confidence, threshold = for_one_image(facepp_client, image, model, age)

        confidences.append(confidence)
        thresholds.append(threshold)

    return np.nanmean(np.asarray(confidences)), np.nanmean(np.asarray(thresholds))


if __name__ == "__main__":
    facepp_client = create_facepp_client('face++_config.ini')
    model_paths = read_caae_paths('face++_config.ini')

    # conf, thresholds = for_one_image(facepp_client, 'metric_images/test.jpg', model_paths['patchgan'], 40)
    # print(conf)
    # print(thresholds["1e-5"])

    print(for_more_images(facepp_client, '../../data/UTKFace', model_paths['patchgan'], 40, 1000))


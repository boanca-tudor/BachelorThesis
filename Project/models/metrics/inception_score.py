from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.efficientnet_v2 import preprocess_input

from models.caae import CAAE
from models.age_classifier import *
import cv2


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)

    return np.asarray(images_list)


def calculate_inception_score(images, n_split=10, eps=1E-16):
    inception_model = create_model(128)
    inception_model.load_weights("../../age_classifier_checkpoints/weights.08.hdf5")
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start: ix_end]

        subset = preprocess_input(subset)
        p_yx = inception_model.predict(subset)

        p_y = expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = mean(sum_kl_d)
        is_score = exp(avg_kl_d)
        scores.append(is_score)

    return mean(scores), std(scores)


if __name__ == "__main__":
    dataset_path = '../../data/UTKFace/'
    model = CAAE(z_channels=100,
                 l_channels=10,
                 gen_channels=1024,
                 dataset_size=len(list_full_paths(dataset_path)))

    checkpoint_dir = '../../2023-05-15/200_epochs_UTKFace/'
    model.load_model(checkpoint_dir)

    ages = create_all_ages()
    image_paths = list_full_paths(dataset_path)
    np.random.shuffle(image_paths)
    image_filenames = image_paths[0:500]
    images = []
    for image in image_filenames:
        images.append(tf.expand_dims(load_image(image), axis=0))
    results = []

    for image in images:
        for age in ages:
            results.append(tf.squeeze(model.eval([image, age])).numpy())

    is_avg, is_std = calculate_inception_score(np.asarray(results), n_split=10)
    print(is_avg, is_std)


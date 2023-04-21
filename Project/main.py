from model import *
from utils import *
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    model = CAAE(100, 10, 1024)
    # dataset_path = 'data/UTKFace/'
    # images = list_full_paths(dataset_path)
    # np.random.shuffle(images)
    # images = np.random.choice(images, 64)
    # x, ages = read_images(images)

    # model.train(50, dataset_path, 64)
    checkpoint_dir = '2023-04-21/50_epochs_UTKFace/'
    model.load_model(checkpoint_dir)

    ages = create_all_ages()
    image = load_image('test.jpg')
    results = [image]
    image = tf.expand_dims(load_image('test.jpg'), axis=0)
    for age in ages:
        results.append(tf.squeeze(model.eval([image, age])).numpy())

    all_images = np.concatenate(results)
    plt.imshow(all_images)
    plt.show()

    # x = tf.expand_dims(x, axis=0)
    # y, dz, dimg = model(x, 50)
    #
    # y = tf.keras.utils.normalize(y)
    # print(y)

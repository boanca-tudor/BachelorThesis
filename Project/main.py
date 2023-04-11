from model import *
from utils import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = CAAE(100, 10, 1024)
    base_dir = 'data/UTKFace/'
    images = list_full_paths(base_dir)
    np.random.shuffle(images)
    images = np.random.choice(images, 64)
    x = read_batch(images)
    fig = plt.figure(figsize=(10, 10))

    index = 1
    for image in x:
        fig.add_subplot(8, 8, index)
        plt.imshow(image.numpy())
        index += 1

    plt.show()

    # x = tf.expand_dims(x, axis=0)
    # y, dz, dimg = model(x, 50)
    #
    # y = tf.keras.utils.normalize(y)
    # print(y)

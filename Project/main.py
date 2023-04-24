from model import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_path = 'data/UTKFace/'
    model = CAAE(z_channels=100,
                 l_channels=10,
                 gen_channels=1024,
                 dataset_size=len(list_full_paths(dataset_path)))
    # training
    # model.train(50, dataset_path, 64)

    # eval
    checkpoint_dir = '2023-04-24/25_epochs_UTKFace/'
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

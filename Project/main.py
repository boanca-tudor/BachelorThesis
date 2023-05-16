from models.caae import *
from model_utils import *
import numpy as np
from skimage.io import imsave


if __name__ == '__main__':
    dataset_path = 'data/UTKFace/'
    model = CAAE(z_channels=100,
                 l_channels=10,
                 gen_channels=1024,
                 dataset_size=len(list_full_paths(dataset_path)))
    # model.train(epochs=150, dataset_path=dataset_path,
    #             batch_size=64,
    #             save="every5",
    #             previous_path='2023-05-14/50_epochs_UTKFace/')

    # eval
    checkpoint_dir = '2023-05-15/200_epochs_UTKFace/'
    model.load_model(checkpoint_dir)

    ages = create_all_ages()
    image = load_image('test.jpg')
    image2 = load_image('test4.jpg')
    images = [image, image2]
    results = []
    all_results = []
    for image in images:
        results = [image]
        for age in ages:
            image_tensor = tf.expand_dims(image, axis=0)
            results.append(tf.squeeze(model.eval([image_tensor, age])).numpy())

        all_results.append(np.concatenate(results))

    all_results = np.asarray(all_results)
    final_result = all_results[0]
    for i in range(1, len(all_results)):
        final_result = np.concatenate((final_result, all_results[i]), axis=1)
    imsave("result.jpg", (((final_result + 1) / 2) * 255).astype(np.uint8))

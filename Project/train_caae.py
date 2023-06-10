from models.caae import *
from models.model_utils import *

if __name__ == '__main__':
    dataset_path = 'data/UTKFace/'
    model = CAAE(z_channels=100,
                 l_channels=10,
                 gen_channels=1024,
                 dataset_size=len(list_full_paths(dataset_path)))
    # training
    model.train(50, dataset_path, 32)


    # # eval
    # checkpoint_dir = '2023-05-22/50_epochs_UTKFace/'
    # model.load_model(checkpoint_dir)
    #
    # ages = create_all_ages()
    # image = load_image('test.jpg')
    # results = [image]
    # image = tf.expand_dims(image, axis=0)
    # for age in ages:
    #     results.append(tf.squeeze(model.eval([image, age])).numpy())
    #
    # results = np.concatenate(results, axis=0)
    # imsave("result.jpg", (((results + 1) / 2) * 255).astype(np.uint8))

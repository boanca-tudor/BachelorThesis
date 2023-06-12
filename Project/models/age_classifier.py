import math
import keras
from keras import layers
from keras_vggface.vggface import VGGFace
from keras.utils import to_categorical
from keras_vggface import utils
from models.model_utils import *
from keras.utils import Sequence
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
import datetime


def create_model(image_size):
    base_model = VGGFace(
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False
    flatten_layer = layers.Flatten()
    dense_1 = layers.Dense(200, activation='relu')
    prediction = layers.Dense(99, activation='softmax')
    return keras.Sequential([
        base_model,
        flatten_layer,
        dense_1,
        prediction
    ])


def calc_age(taken, dob):
    birth = datetime.datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_imdb_meta(mat_path, db="imdb"):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def get_cacd_meta(mat_path):
    meta = loadmat(mat_path)
    image_data = meta['celebrityImageData']

    return image_data[0, 0][0], image_data[0, 0][-1]


def get_from_images(dataset_path):
    ages = []
    images = []
    for file in os.listdir(dataset_path):
        ages.append(int(file.split("_")[0]))
        images.append(file)

    return ages, images


class ImageSequence(Sequence):
    def __init__(self, x, y, dataset_path, batch_size=64):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.indices = np.arange(len(x))
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))

        imgs = []
        ages = []
        for i in self.indices[start: end]:
            image = resize(imread(self.dataset_path + self.x[i]), (128, 128))
            if image.shape == (128, 128, 3):
                imgs.append(utils.preprocess_input(image))
                ages.append(to_categorical(self.y[i], 99))

        return np.asarray(imgs), np.asarray(ages)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008

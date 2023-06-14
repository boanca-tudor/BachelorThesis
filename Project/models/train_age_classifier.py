import numpy as np

from models import age_classifier
from models.age_classifier import *
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
import datetime
from matplotlib import pyplot as plt


def load_dataset():
    full_path, _, _, _, face_score, second_face_score, age = get_imdb_meta("data/imdb_crop/imdb.mat")
    ages = []
    images = []
    sample_num = len(age)
    for i in range(sample_num):
        if face_score[i] < 1.0:
            continue

        if not np.isnan(second_face_score[i]) and second_face_score[i] > 0.0:
            continue

        if not (0 <= age[i] <= 100):
            continue

        images.append(full_path[i][0])
        ages.append(age[i])

    return images, ages


if __name__ == "__main__":
    model = age_classifier.create_model(128)
    BATCH_SIZE = 64
    EPOCHS = 20
    INITIAL_LR = 0.001

    # ages, images = get_from_images("data/MiniCACD/")

    # age, image_paths = get_cacd_meta("data/CACD2000/celebrity2000_meta.mat")

    # images, ages = load_dataset()
    #
    # ages = np.asarray(ages)
    # images = np.asarray(images)
    #
    # np.save("data/imdb/ages", ages)
    # np.save("data/imdb/images", images)

    ages = np.load("data/imdb/ages.npy")
    images = np.load("data/imdb/images.npy")

    X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2)

    train_gen = ImageSequence(X_train, y_train, "data/imdb_crop/", BATCH_SIZE)
    test_gen = ImageSequence(X_test, y_test, "data/imdb_crop/", BATCH_SIZE)

    callbacks = []

    checkpoint_dir = f"age_classifier_checkpoints/{datetime.date.today()}"
    filename = "weights.{epoch:02d}-val_loss-{val_loss:.2f}.hdf5"

    callbacks.extend([
        LearningRateScheduler(schedule=Schedule(EPOCHS, INITIAL_LR)),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=["categorical_crossentropy"],
        metrics=['accuracy']
    )

    # image = load_image("test3.jpg")
    # images = preprocess_input([image])
    # print(np.argmax(model.predict(np.asarray(images))))

    model.load_weights('weights.hdf5')

    history = model.fit(train_gen, verbose=1, epochs=EPOCHS, callbacks=callbacks, validation_data=test_gen)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('age_classifier_metrics')

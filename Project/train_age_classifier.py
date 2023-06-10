from models import age_classifier
from models.age_classifier import *
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.applications.efficientnet_v2 import preprocess_input
import datetime


if __name__ == "__main__":
    model = age_classifier.create_model(128)

    # ages, images = get_from_images("data/MiniCACD/")

    age, image_paths = get_cacd_meta("data/CACD2000/celebrity2000_meta.mat")

    # full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_imdb_meta("data/imdb_crop/imdb.mat")

    ages = []
    images = []
    sample_num = len(age)

    for i in range(sample_num):
        ages.append(age[i][0])
        images.append(image_paths[i][0][0])

    X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

    train_gen = ImageSequence(X_train, y_train, "data/CACD2000/", 64)
    test_gen = ImageSequence(X_test, y_test, "data/CACD2000/", 64)

    callbacks = []

    checkpoint_dir = f"age_classifier_checkpoints/{datetime.date.today()}"
    filename = "weights.{epoch:02d}.hdf5"

    epochs = 8

    callbacks.extend([
        LearningRateScheduler(schedule=Schedule(epochs, 0.001)),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="accuracy",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=["sparse_categorical_crossentropy"],
        metrics=['accuracy']
    )

    model.load_weights("age_classifier_checkpoints/2023-05-05/weights.08.hdf5")

    image = load_image("test3.jpg")
    images = preprocess_input([image])
    print(np.argmax(model.predict(np.asarray(images))))

    # model.fit(train_gen, epochs=epochs, callbacks=callbacks, validation_data=test_gen)

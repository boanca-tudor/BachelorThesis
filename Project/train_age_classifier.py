from models import age_classifier
from models.age_classifier import *
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
import datetime


if __name__ == "__main__":
    model = age_classifier.create_model(128)

    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta("data/imdb_crop/imdb.mat")

    ages = []
    images = []
    sample_num = len(face_score)

    for i in range(sample_num):
        ages.append(age[i])
        images.append(full_path[i][0])

    X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

    train_gen = ImageSequence(X_train, y_train, 64)
    test_gen = ImageSequence(X_test, y_test, 64)

    callbacks = []

    checkpoint_dir = f"age_classifier_checkpoints/{datetime.date.today()}"
    filename = "weights.{epoch:02d}.hdf5"

    callbacks.extend([
        LearningRateScheduler(schedule=Schedule(30, 0.001)),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=["sparse_categorical_crossentropy"],
        metrics=['accuracy']
    )

    model.summary()

    model.fit(train_gen, epochs=30, callbacks=callbacks, validation_data=test_gen)

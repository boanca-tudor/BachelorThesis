from models import age_classifier
from keras.optimizers import Adam

if __name__ == "__main__":
    model = age_classifier.create_model(128)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
        metrics=['accuracy']
    )

    model.summary()
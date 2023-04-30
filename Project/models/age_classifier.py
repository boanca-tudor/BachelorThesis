import keras
from keras import applications, layers


def create_model(image_size):
    base_model = applications.EfficientNetV2B3(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_age = layers.Dense(units=101, activation='softmax', name='pred_age')(features)
    return keras.Model(inputs=base_model.input, outputs=pred_age)


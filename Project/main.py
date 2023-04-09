from model import *
import tensorflow as tf
import cv2

if __name__ == '__main__':
    model = CAAE(100, 10, 1024)
    x = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = tf.expand_dims(x, axis=0)
    x /= 255

    print(x)
    cv2.imshow('image', tf.reshape(x, (128, 128, 3)).numpy())
    cv2.waitKey(0)
    y, dz, dimg = model(x, 50)

    y = tf.keras.utils.normalize(y)
    y = tf.reshape(y, (128, 128, 3))
    print(y)

    cv2.imshow('image', y.numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


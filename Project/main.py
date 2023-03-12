from model import *
import tensorflow as tf

if __name__ == '__main__':
    # print(tf.config.list_physical_devices())
    x = tf.ones((1, 128, 128, 3))
    model = Autoencoder(100, 10, 1024)
    y = model(x, 50)

    print(y)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

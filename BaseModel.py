import keras
import tensorflow as tf
from tensorflow.keras import layers


def kl_divergence(p, q):
    return tf.reduce_sum(p * (tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


class BaseModel(keras.Model):
    def __init__(self, input_dim, classes_count):
        super(BaseModel, self).__init__()
        self.layer1 = layers.Dense(32, activation="relu", name="layer1")
        self.layer2 = layers.Dense(32, activation="relu", name="layer2")
        self.layer3 = layers.Dense(32, activation="relu", name="layer3")
        self.layer4 = layers.Dense(32, activation="relu", name="layer4")
        self.layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")

    def call(self, inputs, training=None, mask=None):
        # x = self.inputs(inputs)
        # x = self.layer1(x)
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)

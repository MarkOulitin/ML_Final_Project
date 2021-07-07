import keras
import tensorflow as tf
from tensorflow.keras import layers


def kl_divergence(p, q):
    return tf.reduce_sum(p * (tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


class BaseModel(keras.Model):
    def __init__(self, input_dim, classes_count):
        super(BaseModel, self).__init__()

        self.inputs = layers.Input(shape=(input_dim,))
        self.layer1 = layers.Dense(32, activation="relu", name="layer1")(self.inputs)
        self.layer2 = layers.Dense(32, activation="relu", name="layer2")(self.layer1)
        self.layer3 = layers.Dense(32, activation="relu", name="layer3")(self.layer2)
        self.layer4 = layers.Dense(32, activation="relu", name="layer4")(self.layer3)
        self.layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")(self.layer4)
        self.model = keras.Model(inputs=self.inputs, outputs=self.layer5)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

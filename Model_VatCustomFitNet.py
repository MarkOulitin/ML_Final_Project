import keras
from keras import layers


class ModelVatCustomFitNet:
    def __init__(self, input_dim, classes_count):
        self.inputs = layers.Input(shape=(input_dim,))
        self.layer1 = layers.Dense(32, activation="relu", name="layer1")(self.inputs)
        self.layer2 = layers.Dense(32, activation="relu", name="layer2")(self.layer1)
        self.layer3 = layers.Dense(32, activation="relu", name="layer3")(self.layer2)
        self.layer4 = layers.Dense(32, activation="relu", name="layer4")(self.layer3)
        self.layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")(self.layer4)
        self.model = keras.Model(inputs=self.inputs, outputs=self.layer5)
        self.model.compile()
        self.inputs = self.model.inputs
        self.outputs = self.model.outputs

    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)

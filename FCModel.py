import keras
import tensorflow as tf
from tensorflow.keras import layers

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.math.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def kl_divergence(y_true, y_pred):
    q = tf.nn.softmax(y_true)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(y_true), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(y_pred), 1))
    return qlogq - qlogp

class CustomModel(keras.Model):
    def __init__(self, method, input_dim, classes_count, epsilon, alpha, xi):
        super(CustomModel, self).__init__()
        self.method = method
        self.inputs = tf.keras.Input(shape=(input_dim,))
        self.layer1 = layers.Dense(32, activation="relu", name="layer1")(self.inputs)
        self.layer2 = layers.Dense(32, activation="relu", name="layer2")(self.layer1)
        self.layer3 = layers.Dense(32, activation="relu", name="layer3")(self.layer2)
        self.layer4 = layers.Dense(32, activation="relu", name="layer4")(self.layer3)
        self.layer5 = layers.Dense(classes_count, activation="relu", name="layer5")(self.layer4)
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.setup_loss(self.inputs, epsilon, alpha, xi)

    def call(self, inputs, training=None, mask=None):
        x = self.inputs(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)

    def get_config(self):
        pass

    def setup_loss(self, inputs, epsilon, alpha, xi):
        self.add_loss(self.build_vat_loss(inputs, epsilon, alpha, xi))

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        self.my_train(x, y)
        if self.method == 'OUR':
            r_vadvs = self.compute_rvadvs(x, y, self.epsilon, self.xi)
            self.my_train(x + r_vadvs, y)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def get_normalized_vector(self, d):
        d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
        return d

    def build_vat_loss(self, inputs, epsilon, alpha, xi):
        def loss(y_true, _):
            r_vadvs = self.compute_rvadvs(inputs, y_true, epsilon, xi)
            y_hat_vadvs = self.call(inputs + r_vadvs)
            R_vadv = kl_divergence(y_true, y_hat_vadvs)
            return alpha * R_vadv

        return loss

    def compute_rvadvs(self, x, y, epsilon, xi):
        d = tf.random.normal(shape=tf.shape(x))
        num_of_iterations = 1
        for _ in range(num_of_iterations):
            d = xi * self.get_normalized_vector(d)
            y_hat = self.call(x + d)
            dist = kl_divergence(y, y_hat)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
        return epsilon * self.get_normalized_vector(d)

    def my_train(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

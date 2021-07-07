import keras
import tensorflow as tf
from tensorflow.keras import layers
from BaseModel import BaseModel
import keras.backend as K
from functools import reduce
from keras.utils.generic_utils import to_list

kl_func_loss = tf.keras.losses.KLDivergence()

def kl_divergence(p, q):
    return kl_func_loss(p,q)
    # return tf.reduce_sum(p * (tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


def get_normalized_vector(d):
    return tf.math.l2_normalize(d)


class CustomModel(BaseModel):
    def __init__(self, method, input_dim, classes_count, epsilon, alpha, xi):
        super(CustomModel, self).__init__(input_dim, classes_count)
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.setup_loss(self.inputs, epsilon, alpha, xi)

    def setup_loss(self, inputs, epsilon, alpha, xi):
        # self.add_loss(self.build_vat_loss(inputs, epsilon, alpha, xi))
        pass

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        self.my_train(x, y)
        # if self.method == 'OUR':
        #     r_vadvs = self.compute_rvadvs(x, y, self.epsilon, self.xi)
        #     self.my_train(x + r_vadvs, y)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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
            d = xi * get_normalized_vector(d)
            y_hat = self.call(x + d)
            dist = kl_divergence(y, y_hat)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
        return epsilon * get_normalized_vector(d)

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

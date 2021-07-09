import time
import keras
from keras import losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

kl_func_loss = tf.keras.losses.KLDivergence()


def kl_divergence(p, q):
    return kl_func_loss(p, q)
    # return tf.reduce_sum(p * (tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


def get_normalized_vector(d):
    return tf.math.l2_normalize(d)


def split_to_batches(x, y, batch_size):
    indexes = np.arange(x.shape[0])
    n_batches = x.shape[0] // batch_size
    end = batch_size * n_batches
    np.random.shuffle(indexes)
    for step, batch_low_index in enumerate(range(0, end, batch_size)):
        batch_indices = indexes[batch_low_index:batch_low_index + batch_size - 1]
        x_batch_train = x[batch_indices, :]
        y_batch_train = y[batch_indices, :]
        yield x_batch_train, y_batch_train


class ModelVatCustomFit(keras.Model):
    def __init__(self, inputs, outputs, method, epsilon, alpha, xi):
        super(ModelVatCustomFit, self).__init__(inputs=inputs, outputs=outputs)
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.cross_entropy = losses.CategoricalCrossentropy()

    def get_config(self):
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=32,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        assert len(x.shape) == 2 and len(y.shape) == 2
        assert x.shape[0] == y.shape[0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        start_time = time.time()
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time_epoch = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(split_to_batches(x, y, batch_size)):
                with tf.GradientTape() as tape:
                    y_pred = self(x_batch_train, training=True)
                    with tape.stop_recording():
                        r_vadvs = self.compute_rvadvs(x_batch_train, y_pred, self.epsilon, self.xi)
                    y_hat_vadvs = self(x_batch_train + r_vadvs, training=False)
                    loss_value = self.compute_loss(y_batch_train, y_pred, y_hat_vadvs)
                grads = tape.gradient(loss_value, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # Update training metric.
                # train_acc_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(f"Seen so far: {step + 1} batches")

            # Display metrics at the end of each epoch.
            # train_acc = train_acc_metric.result()
            # print("Training acc over epoch: %.4f" % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            # train_acc_metric.reset_states()

            print(f"Epoch {epoch}/{epochs} done, loss = {loss_value} took %.2fs" % (time.time() - start_time_epoch))

        print("Time taken: %.2fs" % (time.time() - start_time))

    def compute_loss(self, y_true, y_pred, y_hat_vadvs):
        if self.method == 'OUR':
            R_vadv = kl_divergence(y_true, y_hat_vadvs)
        else:
            R_vadv = kl_divergence(y_pred, y_hat_vadvs)
        return self.cross_entropy(y_true, y_pred) + self.alpha * R_vadv

    def compute_rvadvs(self, x, y, epsilon, xi):
        d = tf.random.normal(shape=tf.shape(x))
        d = tf.Variable(d, True)
        num_of_iterations = 1
        with tf.GradientTape() as d_tape:
            for _ in range(num_of_iterations):
                d = xi * get_normalized_vector(d)
                y_hat = self(x + d)
                dist = kl_divergence(y, y_hat)
                grad = d_tape.gradient(dist, [d])[0]
                d = tf.stop_gradient(grad)
            return epsilon * get_normalized_vector(d)

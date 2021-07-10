import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, metrics

kl_func_loss = losses.KLDivergence()


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
        y_batch_train = y[batch_indices]

        x_batch_train = tf.convert_to_tensor(x_batch_train)
        y_batch_train = tf.convert_to_tensor(y_batch_train)
        yield x_batch_train, y_batch_train


class ModelVatCustomFit(keras.Model):
    def __init__(self, inputs, outputs, method, epsilon, alpha, xi):
        super(ModelVatCustomFit, self).__init__(inputs=inputs, outputs=outputs)
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.train_time = None

    def get_config(self):
        pass

    # code inspired from
    # https://keras.io/guides/writing_a_training_loop_from_scratch
    # https://keras.io/guides/customizing_what_happens_in_fit/
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
        assert len(x.shape) == 2
        assert x.shape[0] == y.shape[0]

        if len(y.shape) == 2:
            train_acc_metric = metrics.CategoricalAccuracy()
        elif len(y.shape) == 1:
            train_acc_metric = metrics.BinaryAccuracy()
        else:
            assert False

        start_time = time.time()
        for epoch in range(epochs):
            # print("\nStart of epoch %d" % (epoch,))
            start_time_epoch = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(split_to_batches(x, y, batch_size)):
                with tf.GradientTape() as tape:
                    y_pred = self(x_batch_train, training=True)
                    if self.method == 'Dropout':
                        loss_value = self.compiled_loss(y_batch_train, y_pred)
                    else:
                        with tape.stop_recording():
                            r_vadvs = self.compute_rvadvs(x_batch_train, y_pred, self.epsilon, self.xi)
                        y_hat_vadvs = self(x_batch_train + r_vadvs, training=False)
                        loss_value = self.compute_loss(y_batch_train, y_pred, y_hat_vadvs)
                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.compiled_metrics.update_state(y, y_pred)

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, y_pred)

                # Log every 200 batches.
                # if step % 200 == 0:
                #     print(f"Seen so far: {step + 1} batches")

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            # print("Training acc over epoch: %.4f" % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # print(
            #     f"Epoch {epoch + 1}/{epochs} "
            #     f"done, loss={loss_value}, "
            #     f"train acc={train_acc * 100:.2f}%, "
            #     f"took {(time.time() - start_time_epoch):.2f}s"
            # )

        self.train_time = time.time() - start_time
        # print("Time taken: %.2fs" % self.train_time)

    def compute_loss(self, y_true, y_pred, y_hat_vadvs):
        if self.method == 'OUR':
            R_vadv = kl_divergence(y_true, y_hat_vadvs)
        else:
            R_vadv = kl_divergence(y_pred, y_hat_vadvs)
        return self.compiled_loss(y_true, y_pred) + self.alpha * R_vadv

    # inspired from https://github.com/takerum/vat_tf/blob/c5125d267531ce0f10b2238cf95604d287de63c8/vat.py#L39
    def compute_rvadvs(self, x, y, epsilon, xi):
        d = tf.random.normal(shape=tf.shape(x))
        num_of_iterations = 1
        for _ in range(num_of_iterations):
            d = tf.Variable(d, True)
            with tf.GradientTape() as d_tape:
                d = xi * get_normalized_vector(d)
                y_hat = self(x + d, training=False)
                dist = kl_divergence(y, y_hat)
                grad = d_tape.gradient(dist, [d])[0]
                d = tf.stop_gradient(grad)
        return epsilon * get_normalized_vector(d)

    # taken from https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/sequential.py#L441
    def predict_classes(self, x, batch_size=32, verbose=0):
        """Generate class predictions for the input samples.
        The input samples are processed batch by batch.
        Args:
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
        Returns:
            A numpy array of class predictions.
        """
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

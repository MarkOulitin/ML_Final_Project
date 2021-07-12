import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, metrics

kl_func_loss = losses.KLDivergence()


def kl_divergence(p, q):
    return kl_func_loss(p, q)


def get_normalized_vector(d):
    """
    Normalizes the input vector using L2 norm
    :param d: Input vector
    :return: Normalized input vector using L2 norm
    """
    return tf.math.l2_normalize(d)


def split_to_batches(x, y, batch_size):
    """
    A generator which yields batches of the input data of size `batch_size`
    :param x: Dataset instances x attributes
    :param y: Dataset target (instances x label)
    (label may or may not be one-hot encoded)
    :param batch_size: The size of each batch
    :return: A generator splitting the input data to batches
    """

    indexes = np.arange(x.shape[0])
    n_batches = x.shape[0] // batch_size
    end = batch_size * n_batches
    np.random.shuffle(indexes)
    for step, batch_low_index in enumerate(range(0, end, batch_size)):
        batch_indices = indexes[batch_low_index:batch_low_index + batch_size]
        x_batch_train = x[batch_indices, :]
        y_batch_train = y[batch_indices]

        # make sure we are working with tensorflow tensors
        # to avoid calculation issues
        x_batch_train = tf.convert_to_tensor(x_batch_train)
        y_batch_train = tf.convert_to_tensor(y_batch_train)
        yield x_batch_train, y_batch_train


class VatKerasModel(keras.Model):
    """
    A custom model to implement a custom fit training loop for
    virtual adversarial training.
    """
    def __init__(self, inputs, outputs, method, epsilon, alpha, xi):
        """
        Initializes the model with the specified input layer, outplay layer,
        training variant and hyper parameters.
        See the article for the meaning of each hyper-parameter.

        :param inputs: The input layer
        :param outputs: The output layer
        :param method: A string representing the kind of algorithm
        used for the training.
        Can be one of the following:
          - 'Article' for the original article behavior
          - 'OUR' for our change suggestion
          - 'Dropout' for a normal fit, no virtual adversarial training
        :param epsilon: epsilon hyper-parameter, see the original article
        :param alpha:   alpha   hyper-parameter, see the original article
        :param xi:      xi      hyper-parameter, see the original article
        """
        super(VatKerasModel, self).__init__(inputs=inputs, outputs=outputs)
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.train_time = None

    def get_config(self):
        pass

    # code adopted from
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
            for step, (x_batch_train, y_batch_train) in enumerate(split_to_batches(x, y, batch_size)):
                with tf.GradientTape() as tape:
                    # forward pass
                    y_pred = self(x_batch_train, training=True)

                    # whether to train with virtual adversarial or not
                    if self.method == 'Dropout':
                        loss_value = self.compiled_loss(y_batch_train, y_pred)
                    else:
                        # calculate the adversarial perturbation, forward pass the
                        # instances with the perturbation and calculate the loss.

                        # Stop recording because we do not want the calculation
                        # of the perturbation to affect the gradients
                        with tape.stop_recording():
                            # calculate the perturbation
                            r_vadvs = self.compute_rvadvs(x_batch_train, y_pred, self.epsilon, self.xi)

                        # forward pass instances near the input using the perturbation
                        y_hat_vadvs = self(x_batch_train + r_vadvs, training=False)
                        loss_value = self.compute_loss(y_batch_train, y_pred, y_hat_vadvs)

                # compute the gradients and apply them
                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.compiled_metrics.update_state(y, y_pred)

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, y_pred)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
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
        """
        Computes the loss as in virtual adversarial training or
        with our change suggestion.
        :param y_true: The actual y labels
        :param y_pred: The y labels the model predicted
        :param y_hat_vadvs: The y labels the model predicted for instances
        with the perturbation
        :return: The computed loss
        """

        # D = kl_divergence
        # R_vadv - the regularization term, see the article
        if self.method == 'OUR':
            R_vadv = kl_divergence(y_true, y_hat_vadvs)
        else:
            R_vadv = kl_divergence(y_pred, y_hat_vadvs)

        # Using the compiled loss as 'fancy' l.
        # See the article for the objective function.
        return self.compiled_loss(y_true, y_pred) + self.alpha * R_vadv

    # code adopted from https://github.com/takerum/vat_tf/blob/c5125d267531ce0f10b2238cf95604d287de63c8/vat.py#L39
    def compute_rvadvs(self, x, y, epsilon, xi):
        """
        Computes the small perturbation of size `epsilon` (called r_vadv).
        See the article for mathematical calculation and proof
        :param x: Input instances
        :param y: Input labels
        :param epsilon: epsilon hyper-parameter
        :param xi: xi hyper-parameter
        :return: The small r_vadv perturbation
        """
        d = tf.random.normal(shape=tf.shape(x))
        num_of_iterations = 1
        for _ in range(num_of_iterations):
            # make sure the tensor is watched by the tape
            d = tf.Variable(d, True)
            with tf.GradientTape() as d_tape:
                d = xi * get_normalized_vector(d)

                # compute the gradient of D by watching the
                # forward pass of the instances with the noise
                # and then calculating D (the probability difference)
                # while watching the calculation.
                y_hat = self(x + d, training=False)
                dist = kl_divergence(y, y_hat)
                grad = d_tape.gradient(dist, [d])[0]

                # make the tape stop watching this particular tensor
                d = tf.stop_gradient(grad)

        # normalize the vector to the maximum allowed size (epsilon)
        return epsilon * get_normalized_vector(d)

    # code adopted from https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/sequential.py#L441
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
            proba = np.squeeze(proba, axis=-1)
            return (proba > 0.5).astype('int32')

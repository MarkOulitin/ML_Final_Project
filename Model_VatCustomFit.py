import time
import keras
from keras import losses
import numpy as np
import tensorflow as tf


class ModelVatCustomFit(keras.Model):
    def __init__(self, inputs, outputs, epsilon, alpha, xi, ip):
        super(ModelVatCustomFit, self).__init__(inputs=inputs, outputs=outputs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.xi = xi
        self.ip = ip

    def call(self, inputs, training=None, mask=None):
        super(ModelVatCustomFit, self).__call__(inputs, training=training, mask=mask)

    def get_config(self):
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
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
        assert len(x.shape) == 2 and len(y.shape) == 2 and y.shape[1] == 1
        assert x.shape[0] == y.shape[0]

        n = x.shape[0]
        n_batches = n // batch_size
        loss_fn = losses.CategoricalCrossentropy()

        start_time = time.time()
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time_epoch = time.time()

            # Iterate over the batches of the dataset.
            for step in range(n_batches):
                indices = np.random.randint(low=0, high=n, size=batch_size)
                x_batch_train = x[indices, :]
                y_batch_train = y[indices, :]

                with tf.GradientTape() as tape:
                    logits = self(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # Update training metric.
                # train_acc_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 64))

            # Display metrics at the end of each epoch.
            # train_acc = train_acc_metric.result()
            # print("Training acc over epoch: %.4f" % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            # train_acc_metric.reset_states()

            print(f"Epoch {epoch}/{epochs} done, took %.2fs" % (time.time() - start_time_epoch))

        print("Time taken: %.2fs" % (time.time() - start_time))

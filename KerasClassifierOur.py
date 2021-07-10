import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper, KerasClassifier

# some code in this file was taken and adopted from
# https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/wrappers/scikit_learn.py

class KerasClassifierOur(KerasClassifier):
    def __init__(self, num_classes, build_fn=None, **kwargs):
        super(KerasClassifierOur, self).__init__(build_fn=build_fn, **kwargs)
        self._estimator_type = "classifier"
        self.num_classes = num_classes
        self.model = None
        self.sk_params['num_classes'] = num_classes

    def fit(self, x, y, **kwargs):
        self._setup_classes()
        y = self._check_y(y)

        # note: This check is not compatible with a general model.
        # In general, knowing whether the loss is categorical or not is required.
        # In our case, we do use a categorical loss.
        if self.num_classes > 2:
            y = to_categorical(y, num_classes=self.num_classes)
        return BaseWrapper.fit(self, x, y, **kwargs)

    def _setup_classes(self):
        self.classes_ = np.arange(self.num_classes)
        self.n_classes_ = self.num_classes

    def _check_y(self, y):
        if len(y.shape) == 2 and y.shape[1] > 1:
            return y
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            return np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

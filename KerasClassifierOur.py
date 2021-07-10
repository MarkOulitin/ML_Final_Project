import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper, KerasClassifier

# some code in this file was taken and adopted from
# https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/wrappers/scikit_learn.py

class KerasClassifierOur(KerasClassifier):
    def __init__(self, num_classes, build_fn=None, **kwargs):
        super(KerasClassifierOur, self).__init__(build_fn=build_fn, **kwargs)
        self.num_classes = num_classes
        self.model = None
        self.sk_params['num_classes'] = num_classes

    def fit(self, x, y, **kwargs):
        self._check_y(y)
        self._setup_classes()
        if self.num_classes > 2:
            y = to_categorical(y)
        return BaseWrapper.fit(self, x, y, **kwargs)

    def _setup_classes(self):
        self.classes_ = np.arange(self.num_classes)
        self.n_classes_ = self.num_classes

    def _check_y(self, y):
        if len(y.shape) == 2 and y.shape[1] > 1:
            pass
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            pass
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

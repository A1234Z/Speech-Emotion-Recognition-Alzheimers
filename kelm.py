
import numpy as np
import random
initial=0
from sklearn.base import BaseEstimator, ClassifierMixin


class KELM (BaseEstimator, ClassifierMixin):

    def __init__(self,
                 hid_num,
                 a=1):
       
        self.hid_num = hid_num
        self.a = a
        self.initial=0

    def _sigmoid(self, x):
        
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _add_bias(self, X):

        return np.c_[X, np.ones(X.shape[0])]

    def _ltov(self, n, label):
        
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        X = self._add_bias(X)

        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))
        
        _H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)

        return self
          

    def predict(self, X):
        
        _H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(_H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])
                

  
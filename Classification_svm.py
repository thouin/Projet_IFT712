#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
import numpy as np

class SVM_Sigmoide_Kernel(BaseEstimator):
    def __init__(self):
        self.model = SVC(kernel='rbf', gamma='scale')

    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        pred = self.model.decision_function(data)
        return -hinge_loss(target, pred, labels=np.arange(1, 4)) # TODO: Add regularisation term

    def entrainement(self, x_train, y_train, x_test, y_test):
        return -self.fit(x_train, y_train).score(x_train, y_train), self.model.score(x_train, y_train), -self.score(x_test, y_test), self.model.score(x_test, y_test)


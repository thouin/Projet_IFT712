#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier
import Classification_adaboost as ca

class bagging(BaseEstimator):
    def __init__(self, estimator=ca.adaboost(), n_estimators=10):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, n_jobs=-1)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        return self.model.score(data, target)

    def entrainement(self, x_train, y_train, x_test, y_test):
        return self.fit(x_train, y_train).score(x_train, y_train), self.score(x_test, y_test)

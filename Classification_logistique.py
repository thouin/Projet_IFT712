#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import numpy as np
import warnings

class Regression_Logistique(BaseEstimator):
    def __init__(self, l2reg=0.0001, lr=0.001, max_iter=50):
        self.l2reg = l2reg
        self.lr = lr
        self.max_iter = max_iter
        self.model = SGDClassifier(loss='log', fit_intercept=False, alpha=l2reg, eta0=lr, max_iter=max_iter)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        y_pred = self.model.predict_proba(data)
        return -log_loss(target, y_pred, labels=np.arange(1, 4))

    def __epoch(self,x_train, y_train, x_valid, y_valid):
        classes = np.arange(1, 4)
        self.model.partial_fit(x_train, y_train, classes)
        # On calcule la loss pour les données de validation et d'entraînement
        y_train_pred = self.model.predict_proba(x_train)
        train_loss = log_loss(y_train, y_train_pred, normalize=True, labels=classes)
        y_valid_pred = self.model.predict_proba(x_valid)
        valid_loss = log_loss(y_valid, y_valid_pred, normalize=True, labels=classes)
        # On calcule la justesse pour les données de validation et d'entraînement
        y_train_pred = np.argmax(y_train_pred, axis=1)+1
        train_accu = (y_train_pred == y_train).mean()
        y_valid_pred = np.argmax(y_valid_pred, axis=1)+1
        valid_accu = (y_valid_pred == y_valid).mean()
        return train_loss, train_accu, valid_loss, valid_accu

    def entrainement(self, x_train, y_train,x_valid,y_valid):
        train_lost_list = []
        valid_loss_list = []
        train_accu_list = []
        valid_accu_list = []
        train_loss, train_accu, valid_loss, valid_accu = self.__epoch(x_train, y_train, x_valid, y_valid)
        train_lost_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_accu_list.append(train_accu)
        valid_accu_list.append(valid_accu)
        best_loss = 1e+10
        iter_no_change = 0
        for i in range(self.max_iter-1):
            train_loss, train_accu, valid_loss, valid_accu = self.__epoch(x_train, y_train, x_valid, y_valid)
            train_lost_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            train_accu_list.append(train_accu)
            valid_accu_list.append(valid_accu)
        return train_lost_list, train_accu_list, valid_loss_list, valid_accu_list

    def prediction(self, x):
        a = self.model.predict(x)
        return a

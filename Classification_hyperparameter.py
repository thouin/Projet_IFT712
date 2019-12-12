#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
import Classification_logistique as cl
import Classification_svm as cs
import Classification_neural_net as cn
import Classification_adaboost as ca
import Classification_bagging as cb
import numpy as np

class HyperparameterSearch:
    def __init__(self, estimator, param_grid):
        self.grid = GridSearchCV(estimator, param_grid, n_jobs=-1)

    def fit(self, x_train, y_train):
        self.grid.fit(x_train, y_train)

    def best_estimator(self):
        return self.grid.best_estimator_

    def best_params(self):
        return self.grid.best_params_

def HyperparameterLogistique(x_train, y_train):
    estimator = cl.Regression_Logistique()
    param_grid = {
        'l2reg' : np.linspace(0.1, 10, 0.1),
        'lr' : np.linspace(0.01, 1, 0.01)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()

def HyperparameterSVM(x_train, y_train):
    estimator = cs.SVM_Sigmoide_Kernel()
    param_grid = {
        'coef' : np.linspace(-1, 1, 0.1)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()

def HyperparameterNeuralNet(x_train, y_train, hidden_layers=(6, 6)):
    estimator = cn.neural_net(hidden_layers)
    param_grid = {
        'activation' : ['logistique', 'relu'],
        'l2reg' : np.linspace(0.1, 10, 0.1),
        'lr' : np.linspace(0.01, 1, 0.01),
        'mu' : np.linspace(0, 1, 0.01)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()

def HyperparameterAdaboost(x_train, y_train):
    estimator = ca.adaboost()
    param_grid = {
        'n_estimators' : np.arange(1, 101),
        'lr' : np.linspace(0.01, 1, 0.01)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()

def HyperparameterBagging(x_train, y_train):
    estimator = cb.bagging()
    adaboost, _ = HyperparameterAdaboost(x_train, y_train)
    neural_net, _ = HyperparameterNeuralNet(x_train, y_train)
    param_grid = {
        'estimator' : [adaboost, neural_net],
        'n_estimator' : np.arange(1, 10),
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()


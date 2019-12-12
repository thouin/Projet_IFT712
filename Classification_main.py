#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from Classification_io import io
import Classification_hyperparameter as ch
import Classification_io as ci

def logistique(x_train, x_test, y_train, y_test):
    print("-------- Application de la régression linéaire --------")
    est, params = ch.HyperparameterLogistique(x_train, y_train)
    train_loss_list, train_accu_list, test_loss_list, test_accu_list = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.plot(train_loss_list, train_accu_list, valid_loss_list, valid_accu_list, 'logistique.png')
    ci.io.print_params(params, 'logistique_param.txt')

def svm(x_train, x_test, y_train, y_test):
    print("-------- Application de SVM avec sigmoide --------")
    est = ch.HyperparameterSVM(x_train, y_train)
    train_loss, train_accu, test_loss, test_accu = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.print_errors(train_loss, train_accu, test_loss, test_accu, 'svm_error.txt')

def neural_net(x_train, x_test, y_train, y_test):
    print("-------- Application d'un réseau de neurone --------")
    est, params = ch.HyperparameterNeuralNet(x_train, y_train, hidden_layers=(6, 6))
    train_loss_list, train_accu_list, test_loss_list, test_accu_list = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.plot(train_loss_list, train_accu_list, test_loss_list, test_accu_list, 'neural_net66.png')
    ci.io.print_params(params, 'neural_net66_param.txt')

    est, params = ch.HyperparameterNeuralNet(x_train, y_train, hidden_layers=(6))
    train_loss_list, train_accu_list, test_loss_list, test_accu_list = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.plot(train_loss_list, train_accu_list, test_loss_list, test_accu_list, 'neural_net66.png')
    ci.io.print_params(params, 'neural_net66_param.txt')

def adaboost(x_train, x_test, y_train, y_test):
    print("-------- Application de adaboost avec un arbre de décision de profondeur un --------")
    est, params = ch.HyperparameterAdaboost(x_train, y_train)
    train_accu, test_accu = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.print_params(params, 'adaboost_param.txt')
    ci.io.print_scores(train_accu, test_accu, 'adaboost_error.txt')

def bagging(x_train, x_test, y_train, y_test):
    print("-------- Application de bagging --------")
    est, params = ch.HyperparameterBagging(x_train, y_train)
    train_accu, test_accu = est.entrainement(x_train, y_train, x_test, y_test)
    ci.io.print_params(params, 'bagging_param.txt')
    ci.io.print_scores(train_accu, test_accu, 'bagging_error.txt')

def main():
    X_train, X_test, Y_train, Y_test = io.getData()
    logistique(X_train, X_test, Y_train, Y_test)
    #svm(X_train, X_test, Y_train, Y_test)
    #neural_net(X_train, X_test, Y_train, Y_test)
    #adaboost(X_train, X_test, Y_train, Y_test)
    #bagging(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()

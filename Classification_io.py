#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

class io():
    @staticmethod
    def getData():
        # Importation des données du dataset 'wine' dans un dataframe nommé: df
        # Renommage des attributs
        df = pd.read_csv("wine.data", header=None)
        df.columns = ["Target","Alcoho","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
        "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
        "OD280/OD315_of_diluted_wines","Proline"]


        # Creation des données d'entrainement et de test
        features = ["Alcoho","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
        "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
        "OD280/OD315_of_diluted_wines","Proline"]
        Y = df.Target
        X = df.loc[:, features]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test, Y_train, Y_test
    
    @staticmethod
    def plot(train_loss_list, train_accu_list, test_loss_list, test_accu_list, filename):
        xdata = np.arange(1, len(train_loss_list) + 1)
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.set_ylabel('loss')
        ax1.plot(xdata, train_loss_list, label='training')
        ax1.plot(xdata, test_loss_list, label='test')
        ax1.legend()
        
        ax2.set_ylabel('accuracy')
        ax2.plot(xdata, train_accu_list, label='training')
        ax2.plot(xdata, test_accu_list, label='test')
        ax2.legend()
        fig.savefig(filename)

    @staticmethod
    def print_params(params, filename):
        with open(filename, 'w') as f:
            for keys,values in params.items():
                f.write(keys + ': ' + str(values) + '\n')

    @staticmethod
    def print_errors(train_loss, train_accu, test_loss, test_accu, filename):
        with open(filename, 'w') as f:
            f.write("Erreur d'entraînement : %.4f\n" % train_loss)
            f.write("Erreur de test : %.4f\n" % test_loss)
            f.write("Précision d'entraînement : %.4f\n" % train_accu)
            f.write("Précision de test : %.4f\n" % test_accu)

    @staticmethod
    def print_scores(train_accu, test_accu, filename):
        with open(filename, 'w') as f:
            f.write("Précision d'entraînement : %.4f\n" % train_accu)
            f.write("Précision de test : %.4f\n" % test_accu)


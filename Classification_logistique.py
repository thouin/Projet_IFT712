from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import numpy as np
import warnings

class Regression_Logistique:
    def __init__(self, l2reg=0.0, lr=0.001, tol=1e-4, max_iter=200):
        print("-------- Application de la régression linéaire --------")
        self.l2reg = l2reg
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.model = SGDClassifier(loss='log', fit_Intercept=False, alpha=l2reg, learning_rate=lr, tol=tol, max_iter=max_iter)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        pred = self.model.predict_proba(data)
        return -log_loss(target, pred, labels=np.arange(3)) # TODO: Add regularisation term

    def __epoch(x_train, y_train, x_test, y_test, num_classes=3):
        classes = np.aranges(num_classes)
        self.model.partial_fit(x_train, y_train, classes)
        # On calcule la loss pour les données de testation et d'entraînement
        y_train_pred = self.model.predict_proba(x_train)
        train_loss = log_loss(y_train, y_train_pred, normalize=True, labels=classes)
        y_test_pred = self.model.predict_proba(x_test)
        test_loss = log_loss(y_test, y_test_pred, normalize=True, labels=classes)
        # On calcule la justesse pour les données de testation et d'entraînement
        t_train_pred = argmax(t_train_pred, axis=1)
        train_accu = (t_train_pred == y_train).mean()
        t_test_pred = argmax(t_test_pred, axis=1)
        test_accu = (t_test_pred == y_test).mean()
        return train_loss, train_accu, test_loss, test_accu

    def entrainement(self, x_train, x_train):
        train_loss_list = []
        test_loss_list = []
        train_accu_list = []
        test_accu_list = []
        train_loss, train_accu, test_loss, test_accu = self.__epoch(x_train, y_train, x_test, y_test)
        train_lost_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_accu_list.append(train_accu)
        test_accu_list.append(test_accu)
        delta_loss = 0
        for i in range(self.max_iter-1):
            train_loss, train_accu, test_loss, test_accu = self.__epoch(x_train, y_train, x_test, y_test)
            train_lost_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_accu_list.append(train_accu)
            test_accu_list.append(test_accu)
            delta_loss = train_loss_list[-1] - train_loss_list[-2]
            if delta_loss < tol:
                break
        if delta_loss >= tol:
            warnings.warn("neural_net: Nombre maximal d'itération atteint")
        return train_loss_list, train_accu_list, test_loss_list, test_accu_list

    def prediction(self, x):
        a = self.model.predict(x)
        return a

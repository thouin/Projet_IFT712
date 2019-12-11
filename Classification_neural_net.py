import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.log_loss import log_loss

class neural_net:
    def __init__(self, activation='relu', l2reg=0.0, lr=0.001, solver='adam', mu=0.9, hidden_layers=(6, 6), tol=1e-4, max_iter=200):
        print("-------- Application d'un réseau de neurone --------")
        self.activation = activation
        self.l2reg = l2reg
        self.lr = lr
        self.solver = solver
        self.mu = mu
        self.hidden_layers = hidden_layers
        self.tol = tol
        self.max_iter = max_iter
        self.model = MLPClassifier(hidden_layers_sizes=hidden_layers, activation=activation, solver=solver, momentum=mu, learning_rate_init=lr, alpha=l2reg)

    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        pred = self.model.predict_proba(data)
        return -log_loss(target, pred, labels=np.arange(3)) # TODO: Add regularisation term

    def __epoch(x_train, y_train, x_valid, y_valid, num_classes=3):
        classes = np.aranges(num_classes)
        self.model.partial_fit(x_train, y_train, classes)
        # On calcule la loss pour les données de validation et d'entraînement
        t_train_pred = self.model.predict_proba(x_train)
        train_loss = log_loss(y_train, y_train_pred, normalize=True, labels=classes)
        t_valid_pred = self.model.predict_proba(x_valid)
        valid_loss = log_loss(y_valid, y_valid_pred, normalize=True, labels=classes)
        # On calcule la justesse pour les données de validation et d'entraînement
        y_train_pred = argmax(t_train_pred, axis=1)
        train_accu = (y_train_pred == y_train).mean()
        y_valid_pred = argmax(t_valid_pred, axis=1)
        valid_accu = (y_valid_pred == y_valid).mean()
        return train_loss, train_accu, valid_loss, valid_accu

    def entrainement(self, x_train, y_train, x_valid, y_valid):
        train_loss_list = []
        valid_loss_list = []
        train_accu_list = []
        valid_accu_list = []
        train_loss, train_accu, valid_loss, valid_accu = self.__epoch(x_train, y_train, x_valid, y_valid)
        train_lost_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_accu_list.append(train_accu)
        valid_accu_list.append(valid_accu)
        delta_loss = 0
        for i in range(self.max_iter-1):
            train_loss, train_accu, valid_loss, valid_accu = self.__epoch(x_train, y_train, x_valid, y_valid)
            train_lost_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            train_accu_list.append(train_accu)
            valid_accu_list.append(valid_accu)
            delta_loss = train_loss_list[-1] - train_loss_list[-2]
            if delta_loss < tol:
                break
        if delta_loss >= tol:
            warnings.warn("neural_net: Nombre maximal d'itération atteint")
        return train_loss_list, train_accu_list, valid_loss_list, valid_accu_list

    def prediction(self, x):
        return self.model.predict(x)

        

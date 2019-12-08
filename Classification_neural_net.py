from numpy import argmax
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.log_loss import log_loss

class neural_net:
    def __init__(self, activation='relu', l2reg=0.0, lr=0.001, solver='adam', mu=0.9, hidden_layers=(6, 6)):
        print("-------- Application d'un réseau de neurone --------")
        self.model = MLPClassifier(hidden_layers_sizes=hidden_layers, activation=activation, solver=solver, momentum=mu, learning_rate_init=lr, alpha=l2reg)

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


        

from sklearn.neural_network import MLPClassifier

class neural_net:
    def __init__(self, activation='relu', l2reg=0.0, lr=0.001, solver='adam', mu=0.9, hidden_layers=(6, 6)):
        print("-------- Application d'un réseau de neurone --------")
        self.model = MLPClassifier(hidden_layers_sizes=hidden_layers, activation=activation, solver=solver, momentum=mu, learning_rate_init=lr, alpha=l2reg)
        

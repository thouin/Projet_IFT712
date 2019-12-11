from sklearn.ensemble import AdaBoostClassifier

class adaboost:
    def __init__(self, n_estimators, lr):
        self.model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=lr)

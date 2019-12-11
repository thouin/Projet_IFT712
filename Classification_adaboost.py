from sklearn.ensemble import AdaBoostClassifier

class adaboost:
    def __init__(self, n_estimators, lr):
        self.n_estimators = n_estimators
        self.lr = lr
        self.model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=lr)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        return self.model.score(data, target)
    
    def entrainement(self, x_train, y_train):
        return self.fit(y_train, y_train).score(x_train, y_train)

    def prediction(self, x):
        a = self.model.predict(x)
        return a

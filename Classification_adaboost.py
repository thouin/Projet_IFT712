from sklearn.ensemble import AdaBoostClassifier

class adaboost:
    def __init__(self, n_estimators, lr):
        self.model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=lr)
        
    def entrainement(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def prediction(self, X_test):
        a = self.model.predict(X_test)
        return a

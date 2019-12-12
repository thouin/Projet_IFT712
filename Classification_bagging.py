from sklearn.ensemble import BaggingClassifier
import Classification_adaboost as ca

class bagging:
    def __init__(self, estimator=ca.adaboost, n_estimators=10):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_sample = max_sample
        self.model = BaggingClassifier(estimator=estimator, n_estimators=n_estimators, n_jobs=-1)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        return self.model.score(data, target)

    def entrainement(self, x_train, y_train):
        return self.fit(x_train, y_train).score(x_train, y_train)

    def prediction(self, x):
        a = self.model.predict(x)
        return a

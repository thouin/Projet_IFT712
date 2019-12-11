from sklearn.ensemble import BaggingClassifier

class bagging:
    def __init__(self, estimator, n_estimators, max_sample):
        self.model = BaggingClassifier(estimator=estimator, n_estimators=n_estimators, max_sample=max_sample, n_jobs=-1)
        
    def entrainement(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def prediction(self, X_test):
        a = self.model.predict(X_test)
        return a

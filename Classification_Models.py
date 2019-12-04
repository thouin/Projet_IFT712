from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class SVM_Sigmoide_Kernel:
    def __init__(self):
        print("-------- Application de SVM avec sigmoide --------")
        self.model = SVC(kernel='sigmoid',gamma='scale')
        self.times = 0

    def entrainement(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def prediction(self, X_test):
        a = self.model.predict(X_test)
        return a

    @staticmethod
    def erreur(Y_test, Y_predict):
        err_rate = (Y_predict != Y_test).mean()
        return err_rate


class Regression_Logistique:
    def __init__(self):
        print("-------- Application de la régression linéaire --------")
        self.model = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)
        self.times = 0

    def entrainement(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def prediction(self, X_test):
        a = self.model.predict(X_test)
        return a

    @staticmethod
    def erreur(Y_test, Y_predict):
        err_rate = (Y_predict != Y_test).mean()
        return err_rate
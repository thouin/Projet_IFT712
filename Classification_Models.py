from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
from sklearn.metrics import log_loss
from numpy import arange

class SVM_Sigmoide_Kernel:
    def __init__(self):
        print("-------- Application de SVM avec sigmoide --------")
        self.model = SVC(kernel='sigmoid',gamma='scale')

    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        pred = self.model.descision_function(data)
        return -hinge_loss(target, pred, labels=np.arange(3)) # TODO: Add regularisation term

    def entrainement(self, x_train, x_train):
        return self.fit(x_train, y_train).score(x_train, y_train), self.model.score(x_train, t_train)

    def prediction(self, x):
        a = self.model.predict(x)
        return a


class Regression_Logistique:
    def __init__(self):
        print("-------- Application de la régression linéaire --------")
        self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=10000)
        
    def fit(self, data, target):
        self.model.fit(data, target)
        return self
        
    def score(self, data, target):
        pred = self.model.predict_proba(data)
        return -log_loss(target, pred, labels=np.arange(3)) # TODO: Add regularisation term

    def entrainement(self, x_train, x_train):
        self.fit(x_train, y_train).score(x_train, y_train), self.model.score(x_train, t_train)

    def prediction(self, x):
        a = self.model.predict(x)
        return a
    

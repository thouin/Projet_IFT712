from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
import numpy as np

class SVM_Sigmoide_Kernel:
    def __init__(self, coef):
        print("-------- Application de SVM avec sigmoide --------")
        self.coef = coef
        self.model = SVC(kernel='sigmoid', gamma='scale', coef0=coef)

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

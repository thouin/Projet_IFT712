from sklearn.model_selection import GridSearchCV
import Classification_logistique as cl
import Classification_svm as cs
import numpy as np

class HyperparameterSearch:
    def __init__(estimator, param_grid):
        self.grid = GridSearchCV(estimator, param_grid, n_jobs=-1)

    def fit(x_train, y_train):
        self.grid.fit(x_train, y_train)

    def best_estimator():
        return self.grid.best_estimator_

    def best_params():
        return self.grid.best_params_

def HyperparameterLogistique(x_train, y_train):
    estimator = cl.Regression_Logistique()
    param_grid = {
        'l2reg' : np.linspace(0.1, 10, 0.1),
        'lr' : np.linspace(0.01, 1, 0.01)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()

def HyperparameterSVM(x_train, y_train):
    estimator = cs.SVM_Sigmoide_Kernel()
    param_grid = {
        'coef' : np.linspace(-1, 1, 0.1)
    }
    search = HyperparameterSearch(estimator, param_grid)
    return search.best_estimator(), search.best_params()


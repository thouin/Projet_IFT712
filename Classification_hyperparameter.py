from sklearn.model_selection import GridSearchCV

class HyperparameterSearch:
    def __init__(estimator, param_grid):
        self.grid = GridSearchCV(estimator, param_grid, n_jobs=-1)

    def fit(x_train, y_train):
        self.grid.fit(x_train, y_train)

    def best_estimator():
        return self.grid.best_estimator_

    def best_params():
        return self.grid.best_params_

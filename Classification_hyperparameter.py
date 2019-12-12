from sklearn.model_selection import GridSearchCV

class HyperparameterSearch:
    def __init__(estimator, param_grid):
        self.grid = GridSearchCV(estimator, param_grid, n_jobs=-1)	

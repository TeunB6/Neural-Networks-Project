from typing import Any
from torch.nn import Module
from itertools import product
from sklearn.model_selection import KFold
import numpy as np

class GridSearch:
    def __init__(self, model: Module, param_grid: dict, folds: int) -> None:
        self.model_class = model
        self.param_combinations = [zip(param_grid.keys(), v) for v in product(*param_grid.keys())]
        self.folds = folds
    
    def __call__(self, X: np.ndarray, y: np.ndarray) -> tuple[Module, int]:
        results = []
        for param in self.param_combinations:
            kFold=KFold(n_splits=self.folds,shuffle=True)
            trial_model = self.model_class(**param)
            scores = []
            for train_index,test_index in kFold.split(X):                
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
                trial_model.fit(X_train, y_train)
                scores.append(trial_model.score(X_test, y_test))
            
            mean_score = np.mean(scores)
            results.append(trial_model, mean_score)
        
        final_model, final_score = max(results, key=lambda x: x[1])
        return final_model, final_model
        


                            

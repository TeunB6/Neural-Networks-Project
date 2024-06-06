from typing import Any
from torch.nn import Module
from itertools import product
from sklearn.model_selection import KFold
import numpy as np


class GridSearch:
    def __init__(self, model, param_grid: dict, folds: int, verbose: int = 0) -> None:
        self.model_class = model
        self.param_combinations = [zip(param_grid.keys(), v) for v in product(*param_grid.values())]
        self.folds = folds
        self.verbose = verbose
    
    def __call__(self, X: np.ndarray, y: np.ndarray) -> tuple[Module, int, dict]: # TODO: Chagne the type for Module to ABC of musicmodel
        results = []
        if self.verbose > 0:
            print(f"Running grid search on {len(self.param_combinations)} combinations of {len(self.param_combinations[0])} parameters...")
        for param in self.param_combinations:
            
            
            kFold=KFold(n_splits=self.folds,shuffle=True)
            trial_model = self.model_class(**param)
            scores = []
            for fold, train_index,test_index in enumerate(kFold.split(X)):                
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
                trial_model.fit(X_train, y_train)
                s = trial_model.score(X_test, y_test)
                
                if self.verbose > 2:
                    print(f"Fold {fold}/{self.folds} - score: {s}")
                
                scores.append(s)
            
            mean_score = np.mean(scores)
            results.append(trial_model, mean_score, param)
            if self.verbose > 1:
                print(f"parameters: {param}  - score: {mean_score}")
        
        final_model, final_score, final_param = max(results, key=lambda x: x[1])
        
        if self.verbose > 0:
            print(f"Selected Parameters: {final_param} - score: {final_score}")
        
        return final_model, final_score, final_param
        


                            

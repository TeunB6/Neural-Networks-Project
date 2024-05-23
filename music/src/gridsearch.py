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
    
    def __call__(self, X_train: np.ndarray, y_train: np.ndarray) -> Module:
        for param in self.param_combinations:
            trial_model = self.model_class(**param)
            for X_fold, y_fold 
                

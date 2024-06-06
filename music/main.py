import os
import pickle as pkl
import numpy as np
from itertools import combinations
from src.utils.preprocess import one_hot_encode
from src.gridsearch import GridSearch
from src.voice_loader import VoiceLoader
from src.music_model import MusicModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

path = "/home/teunb/Assignments/Yr2 - Assignments/Neural-Networks-Project/music/data/F.txt"
def extract_data() -> tuple[list[str], list[str]]:
    size_min = 50 # min is 2 : large, small and crazy were all trained with size_threshold = 2
    size_max = 100
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.split())))
    
    dir = os.path.dirname(path)     
    data = np.asarray(data)[:, 0]
    print("loading data")
    X, y = [], []
    sub_seqs = [data[i:j] for i, j in combinations(range(len(data) + 1), r=2) if size_min <= abs(i-j) <= size_max] 
    for sub in sub_seqs:
        X.append(np.array([one_hot_encode(key) for key in sub[:-1]]))
        y.append(one_hot_encode(sub[-1]), sub[-1])
    
    print("saving data")
    with open(os.path.join(dir, 'X.data'), 'wb') as f: 
        pkl.dump(X, f)
    with open(os.path.join(dir, 'y.data'), 'wb') as f:
        pkl.dump(y, f)

    return X, y

def load_data() -> tuple[list[str], list[str]]:
    dir = os.path.dirname(path)
    with open(os.path.join(dir, 'X.data'), 'rb') as f: 
        X = pkl.load(f)
    with open(os.path.join(dir, 'y.data'), 'rb') as f:
        y = pkl.load(f)
    return X, y
    
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    param_grid = {'' :,
                  '' :,
                  '' :}
    gr = GridSearch(MusicModel, param_grid, 4, verbose=2)
    model = MusicModel(verbose=2)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    model.save("./models", "test")
    
if __name__ == "__main__":
    X, y = load_data()
    train(X, y)
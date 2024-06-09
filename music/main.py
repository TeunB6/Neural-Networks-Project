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
def extract_data():
    size_min = 10 # min is 2 : large, small and crazy were all trained with size_threshold = 2
    size_max = 200
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.split())))
    
    dir = os.path.dirname(path)     
    data = np.asarray(data)[:, 0]
    
    with open("data.txt", "w") as f:
        f.write(str(list(data)))

    
    start_idx = 0
    curr_key = data[0]
    
    sequence = []
    
    for idx, key in enumerate(data):
        if key == curr_key and idx < len(data) - 1:
            continue
        sequence.append((curr_key, idx - start_idx))
        start_idx = idx
        curr_key = key
    
    print("loading data")
    with open("sequence.txt", "w") as f:
        f.write(str(sequence))
    X, y = [], []
    sub_seqs = [sequence[i1:i2] for i1, i2 in combinations(range(len(sequence) + 1), r=2) if size_min <= i2 - i1 <= size_max] 
    for sub in sub_seqs:
        X.append(np.array([np.concatenate((one_hot_encode(key), [num])) for key, num in sub[:-1]]))
        y.append(np.concatenate((one_hot_encode(sub[-1][0]), [sub[-1][1]])))
    print("saving data")
    with open(os.path.join(dir, 'X.data'), 'wb') as f: 
        pkl.dump(X, f)
    with open(os.path.join(dir, 'y.data'), 'wb') as f:
        pkl.dump(y, f)

    return X, y

def load_data():
    dir = os.path.dirname(path)
    with open(os.path.join(dir, 'X.data'), 'rb') as f: 
        X = pkl.load(f)
    with open(os.path.join(dir, 'y.data'), 'rb') as f:
        y = pkl.load(f)
    return X, y
    
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # param_grid = {'' :,
    #               '' :,
    #               '' :}
    # gr = GridSearch(MusicModel, param_grid, 4, verbose=2)
    model = MusicModel(verbose=2)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    model.save("./models", "test")
    
if __name__ == "__main__":
    X, y = load_data()
    train(X, y)
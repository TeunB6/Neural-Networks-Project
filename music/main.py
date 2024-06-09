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

from src.constants import SEQ_LEN_MIN, SEQ_LEN_MAX

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
def extract_data():
    data = []
    with open(os.path.join(path, 'F.txt'), 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.split())))
    
    data = np.asarray(data)[:, 0]
    
    with open(os.path.join(path, "data.txt"), "w") as f:
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
    with open(os.path.join(path, "sequence.txt"), "w") as f:
        f.write(str(sequence))
    X, y = [], []
    sub_seqs = [sequence[i1:i2] for i1, i2 in combinations(range(len(sequence) + 1), r=2) if SEQ_LEN_MIN <= i2 - i1 <= SEQ_LEN_MAX] 
    for sub in sub_seqs:
        X.append(np.array([np.concatenate((one_hot_encode(key), [num])) for key, num in sub[:-1]]))
        y.append(np.concatenate((one_hot_encode(sub[-1][0]), [sub[-1][1]])))
    print("saving data")
    with open(os.path.join(path, 'X.data'), 'wb') as f: 
        pkl.dump(X, f)
    with open(os.path.join(path, 'y.data'), 'wb') as f:
        pkl.dump(y, f)

    return X, y

def load_data():
    with open(os.path.join(path, 'X.data'), 'rb') as f: 
        X = pkl.load(f)
    with open(os.path.join(path, 'y.data'), 'rb') as f:
        y = pkl.load(f)
    return X, y
    
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    # param_grid = {'optimizer_args' : [
    #                                   {"lr" : 0.1, "weight_decay" : 10},
    #                                   {"lr" : 0.01, "weight_decay" : 10},
    #                                   {"lr" : 0.001, "weight_decay" : 10},
    #                                   {"lr" : 0.0001, "weight_decay" : 10},
    #                                   {"lr" : 0.1, "weight_decay" : 100},
    #                                   {"lr" : 0.01, "weight_decay" : 100},
    #                                   {"lr" : 0.001, "weight_decay" : 100},
    #                                   {"lr" : 0.0001, "weight_decay" : 100}],
    #               'hidden_size' : [64, 128],
    #               'num_layers' : [1,4,16]}
    # gr = GridSearch(MusicModel, param_grid, 2, verbose=2)
    # model, score, parameters = gr(X_train, y_train)
    # print(score, parameters)
    
    model = MusicModel(verbose=2)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    model.save("./models", "test")
    
if __name__ == "__main__":
    try:
        X, y = load_data()
    except FileNotFoundError:
        X, y = extract_data()
    train(X, y)
import os
import torch
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


def create_subsequences(sequence):
    return [sequence[i1:i2] for i1, i2 in combinations(range(len(sequence) + 1), r=2) if SEQ_LEN_MIN <= i2 - i1 <= SEQ_LEN_MAX] 

def create_dataset(sequence):
    X, y = [], []
    sub_seqs = create_subsequences(sequence)
    for sub in sub_seqs:
        X.append(np.array([np.concatenate((one_hot_encode(key), [num])) for key, num in sub[:-1]]))
        y.append(np.concatenate((one_hot_encode(sub[-1][0]), [sub[-1][1]])))
    return X, y

def extract_data():
    data = []
    with open(os.path.join(path, 'F.txt'), 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.split())))
    
    data = np.asarray(data)[:, 0]
        
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
        for note, num in sequence:
            f.write(f'{note} {num}\n')


    X, y = create_dataset(sequence)
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
    
def train():
    try:
        X, y = load_data()
    except FileNotFoundError:
        X, y = extract_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    model = MusicModel(verbose=3, epochs=100, batch_size=20)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    return model

def grid():  
    try:
        X, y = load_data()
    except FileNotFoundError:
        X, y = extract_data() 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
 
    param_grid = {'optimizer' : [torch.optim.SGD, torch.optim.Adam],
                'optimizer_args' : [{"lr" : 0.1},
                                    {"lr" : 0.01},
                                    {"lr" : 0.001},
                                    {"lr" : 0.0001},
                                    {"lr" : 0.1, "weight_decay" : 1},
                                    {"lr" : 0.01, "weight_decay" : 1},
                                    {"lr" : 0.001, "weight_decay" : 1},
                                    {"lr" : 0.0001, "weight_decay" : 1},
                                    {"lr" : 0.1, "weight_decay" : 10},
                                    {"lr" : 0.01, "weight_decay" : 10},
                                    {"lr" : 0.001, "weight_decay" : 10},
                                    {"lr" : 0.0001, "weight_decay" : 10}],
                'hidden_size' : [64, 128],
                'num_layers' : [1,4,16],
                'batch_size' : [5,10,20,50],
                'epochs' : [1,5,10]}
    gr = GridSearch(MusicModel, param_grid, 2, verbose=2)
    model, score, parameters = gr(X_train, y_train)
    return model

def split_train() -> MusicModel:
    num_splits = 10
    
    # Load data from Sequence.txt
    with open(os.path.join(path, 'sequence.txt')) as file:
        sequence = [tuple(map(int, line.split())) for line in file.readlines()]
    
    seq_len = len(sequence)
    split_size = seq_len // num_splits
    
    # Create list of datasets based on different parts of the series
    data_sets = [create_dataset(sequence[start:end]) for start, end in zip(range(0, seq_len, split_size),
                                                           range(split_size, seq_len, split_size))]
    
    model = MusicModel(verbose=1, epochs=50, batch_size=1)
    
    # Train model on every part of the sequence
    for X, y in data_sets:
        # X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=True)
        model.fit(X, y)
        model.score(X, y)
        model.lr = model.lr 
    
    return model

def train_new(name):
    
    model = grid()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(dir_path, "models/")
    model.save(save_path, name)

def run_existing(name):
    parameters = {'optimizer': torch.optim.Adam, 'optimizer_args': {'lr': 0.0001, 'weight_decay': 10}, 'hidden_size': 128, 'num_layers': 16, 'batch_size': 50}, {'optimizer': <class 'torch.optim.adam.Adam'>, 'optimizer_args': {'lr': 0.01}, 'hidden_size': 64, 'num_layers': 4, 'batch_size': 50, 'epochs': 5}
    model = MusicModel(**parameters)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load(os.path.join(dir_path, "models/"), name)

    X = np.concatenate((one_hot_encode(0), [5]))
    pred = model.predict(X, 100)
    print(pred)


if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) < 2 or len(args) > 3:
        raise ValueError("Invalid command line run: python main.py train | run name")    
    if args[1] == 'train':
        train_new(args[2])
    elif args[1] == 'run':
        run_existing(args[2])
    else:
        raise ValueError("Invalid command line run: python main.py train | run name")
    
    
    
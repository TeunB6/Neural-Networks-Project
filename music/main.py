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
from src.constants import SEQ_LEN_MIN, SEQ_LEN_MAX, INPUT_SIZE

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def create_subsequences(sequence):
    return [sequence[i1:i2] for i1, i2 in combinations(range(len(sequence) + 1), r=2) if SEQ_LEN_MIN <= i2 - i1 <= SEQ_LEN_MAX] 

def create_dataset(sequence):
    X, y = [], []
    sub_seqs = create_subsequences(sequence)
    for sub in sub_seqs:
        padding = [np.zeros((INPUT_SIZE))] * (SEQ_LEN_MAX - len(sub))
        X.append(np.array(padding + [np.concatenate((one_hot_encode(key), [num])) for key, num in sub[:-1]]))
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
    
def train_default():
    try:
        X, y = load_data()
    except FileNotFoundError:
        X, y = extract_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    model = MusicModel(loss_function=torch.nn.CrossEntropyLoss(), verbose=3)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    return model

def train_grid():  
    try:
        X, y = load_data()
    except FileNotFoundError:
        X, y = extract_data() 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
 
    param_grid = {'optimizer' : [torch.optim.SGD],
                'optimizer_args' : [{"lr" : 0.1, "momentum" : 0.9},
                                    {"lr" : 0.01, "momentum" : 0.9},
                                    {"lr" : 0.001, "momentum" : 0.9},
                                    {"lr" : 0.0001, "momentum" : 0.9},
                                    {"lr" : 0.1, "momentum" : 0.5},
                                    {"lr" : 0.01, "momentum" : 0.5},
                                    {"lr" : 0.001, "momentum" : 0.5},
                                    {"lr" : 0.0001, "momentum" : 0.5},
                                    {"lr" : 0.1, "momentum" : 0.09},
                                    {"lr" : 0.01, "momentum" : 0.09},
                                    {"lr" : 0.001, "momentum" : 0.09},
                                    {"lr" : 0.0001, "momentum" : 0.09},
                                    {"lr" : 0.5, "momentum" : 0.9},
                                    {"lr" : 0.05, "momentum" : 0.9},
                                    {"lr" : 0.005, "momentum" : 0.9},
                                    {"lr" : 0.0005, "momentum" : 0.9},
                                    {"lr" : 0.5, "momentum" : 0.5},
                                    {"lr" : 0.05, "momentum" : 0.5},
                                    {"lr" : 0.005, "momentum" : 0.5},
                                    {"lr" : 0.0005, "momentum" : 0.5},
                                    {"lr" : 0.5, "momentum" : 0.09},
                                    {"lr" : 0.05, "momentum" : 0.09},
                                    {"lr" : 0.005, "momentum" : 0.09},
                                    {"lr" : 0.0005, "momentum" : 0.09}
                                    ],
                'hidden_size' : [64],
                'num_layers' : [2],
                'batch_size' : [300]}
    gr = GridSearch(MusicModel, param_grid, 4, verbose=3)
    model, score, parameters = gr(X_train, y_train)
    model.score(X_test, y_test)
    return model

def train_split() -> MusicModel:
    num_splits = 20
    
    # Load data from Sequence.txt
    with open(os.path.join(path, 'sequence.txt')) as file:
        sequence = [tuple(map(int, line.split())) for line in file.readlines()]
    
    seq_len = len(sequence)
    split_size = seq_len // num_splits
    print(f"Split length of {split_size}")
    
    # Create list of datasets based on different parts of the series
    data_sets = [create_dataset(sequence[start:end]) for start, end in zip(range(0, seq_len, split_size),
                                                           range(split_size, seq_len, split_size))]
    
    model = MusicModel(verbose=1, optimizer=torch.optim.SGD, optimizer_args={'lr' : 0.05, "momentum" : 0.9}, epochs=30, hidden_size=256, num_layers=1, batch_size=1)
    
    h = []
    # Train model on every part of the sequence
    for X, y in data_sets[:1]:
        # X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=True)
        model.fit(X, y)
        h.append(model.score(X, y))
        model.lr = model.lr * 0.9
    print(h)
    return model

def train_new(name):
    
    model = train_default()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(dir_path, "models/")
    model.save(save_path, name)

def run_existing(name):
    model = MusicModel(loss_function=torch.nn.CrossEntropyLoss(), verbose=3, epochs=1, batch_size=100)
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
    
    
    
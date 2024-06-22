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

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "models/")


def load_sequence():
    with open(os.path.join(data_path, 'sequence.txt')) as file:
        sequence = [tuple(map(int, line.split())) for line in file.readlines()]

    sequence = [np.concatenate((one_hot_encode(s[0]), [s[1]])) for s in sequence]
    
    return sequence

def load_data():
    with open(os.path.join(data_path, 'X.data'), 'rb') as f: 
        X = pkl.load(f)
    with open(os.path.join(data_path, 'y.data'), 'rb') as f:
        y = pkl.load(f)
    return X, y

def save_data(X, y):
    with open(os.path.join(data_path, 'X.data'), 'wb') as f:
        pkl.dump(X, f)
    with open(os.path.join(data_path, 'y.data'), 'wb') as f:
        pkl.dump(y, f)
        

def train_new(name):
    
    model = MusicModel(verbose = 1)
    
    if os.path.exists(os.path.join(data_path, "X.data")):
        X, y = load_data()
    else:
        sequence = load_sequence()
        X, y = model.sample_reservoir(sequence)
        save_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit_ffn(X_train, y_train)
    
    model.score_ffn(X_test, y_test)
    
    model.save(save_path, name)

def run_existing(name):
    model = MusicModel(loss_function=torch.nn.CrossEntropyLoss(), verbose=3,
                       epochs=1, batch_size=100)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load(os.path.join(dir_path, "models/"), name)

    X = np.concatenate((one_hot_encode(0), [5]))
    X = np.expand_dims(X, 0)
    pred = model.predict(X, 100)
    
    sequence = load_sequence()
    pred = model.predict(sequence, 100)
    
    print(pred)


if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) < 2 or len(args) > 3:
        raise ValueError("Invalid command line, run: python main.py train | run name")
    if args[1] == 'train':
        train_new(args[2])
    elif args[1] == 'run':
        run_existing(args[2])
    else:
        raise ValueError("Invalid command line, run: python main.py train | run name")
    
    
    
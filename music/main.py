import sys
import os
import torch
import pickle as pkl
import numpy as np
from itertools import product
from src.utils.preprocess import one_hot_encode
from src.voice_loader import VoiceLoader
from src.music_model import MusicModel
from sklearn.model_selection import train_test_split

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "models/")
music_path = os.path.join(dir_path, "wav output/")


def load_sequence() -> list[np.ndarray]:
    """Loads the preprocessed sequence from the disk

    Returns:
       list[np.ndarray] : the timeseries saved on the disk as a list of np.ndarrays
    """    
    with open(os.path.join(data_path, 'sequence.txt')) as file:
        sequence = [tuple(map(int, line.split())) for line in file.readlines()]

    sequence = [np.concatenate((one_hot_encode(s[0]), [s[1]])) for s in sequence]
    
    return sequence

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sampled reservoir data from the disk using pickle

    Returns:
        tuple[np.ndarray, np.ndarray] : A tuple of the data and labels as saved on the disk utilised in the last run of the model
    """    
    with open(os.path.join(data_path, 'X.data'), 'rb') as f: 
        X = pkl.load(f)
    with open(os.path.join(data_path, 'y.data'), 'rb') as f:
        y = pkl.load(f)
    return X, y

def save_data(X: np.ndarray, y: np.ndarray) -> None:
    """Saves the data and labels (from the reservoir) provided on the disk using pickle

    Args:
        X (np.ndarray): data    
        y (np.ndarray): labels
    """    
    with open(os.path.join(data_path, 'X.data'), 'wb') as f:
        pkl.dump(X, f)
    with open(os.path.join(data_path, 'y.data'), 'wb') as f:
        pkl.dump(y, f)
        

def train_new(name: str) -> None:
    """
    Train a new model saved under the provided name

    Args:
        name (str): name of the model
    """    
    model = MusicModel(verbose = 1)
    
    # if os.path.exists(os.path.join(data_path, "X.data")):
    #     X, y = load_data()
    # else:
    #     sequence = load_sequence()
    #     X, y = model.sample_reservoir(sequence)
    #     save_data(X, y)
    sequence = load_sequence()
    X, y = model.sample_reservoir(sequence)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit_ffn(X, y) # NOTE: SWAP THIS BACK
    
    model.score_ffn(X_test, y_test)
    
    model.save(save_path, name)
    
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    # lr = LogisticRegression()
    # lr.fit(X, y[:, :-1])
    # y_pred = lr.predict(X_test)
    # accuracy_score(y_test[:, :-1], y_pred)

def grid_search(name: str) -> None:
    """
    Run a grid search of the parameter grid specified below, the models and files are saved in the gridsearch folder in a folder named 'name'. Here the parameters used are stored in a txt file, the model is saved and a sample of the models sound is stored

    Args:
        name (str): the name to save the gridsearch under
    """
    sequence = load_sequence()
    gs_path = os.path.join(dir_path, f'gridsearch_raw/{name}/')
    if not os.path.exists(gs_path):
        os.mkdir(gs_path)
        
    def get_combinations(param_grid: dict) -> list[dict]:
        return [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
    # Generate list of all combinations to try  
    # optimizer_args_grid = {"lr" : [0.1,0.01,0.001,0.0001,0.00001], "weight_decay" : [0,0.1,0.01,1]}
    # optimizer_combinations = get_combinations(optimizer_args_grid)
    param_grid = {"hidden_size" : [256, 512, 1024, 2048],
                  "spectral_radius" : [1.5, 1.3, 1.1, 0.9, 0.7, 0.5, -1],
                  "density" : [1, 0.9, 0.5, 0.1, 0.01, 0.001],
                  "leakage_rate" : [0.9, 0.5, 0.1, 0.01, 0.001]
                  }
    
    voice_loader = VoiceLoader()
    param_combinations = get_combinations(param_grid)
    num_combinations = len(param_combinations)
        
    print(f"Running grid search on {num_combinations} combinations of {len(param_combinations[0])} parameters...")
    for i, param in enumerate(param_combinations):
        try:
            
            if "durr_optimizer" in param.keys():
                param["prob_optimizer"] = param["durr_optimizer"]
                param["prob_optimizer_args"] = param["durr_optimizer_args"]
            elif "prob_optimizer" in param.keys():
                param["durr_optimizer"] = param["prob_optimizer"]
                param["durr_optimizer_args"] = param["prob_optimizer_args"]
            
            # Initialize path and model
            current_path = os.path.join(gs_path, f'{i+1}/')
            os.mkdir(current_path)
            current_model = MusicModel(**param)
            print(f"Currently running {i+1}/{num_combinations}: {param}")
            
            # Collect Data
            X, y = current_model.sample_reservoir(sequence)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Training Model
            current_model.fit_ffn(X_train, y_train) # NOTE: SWAP THIS BACK TO TRAINING DATA
            
            # Score/predictions
            l, a = current_model.score_ffn(X_test, y_test)            
            #NOTE: ADDED LOSS CRITERIUM HERE 
            # Save results
            with open(os.path.join(current_path, "summary.txt"), 'w') as f:
                f.write(f"Parameters used:\n{param}\n")
                f.write(f"'Testing' loss: prob={l[0]}, durr={l[1]} \t 'Testing' Accuracy {a}\n")
                if l[1] < 10:
                    pred_start = current_model.predict(sequence[:50], 400)
                    pred_end = current_model.predict(sequence, 400)
                    f.write(f"predictions for the start of the song:\n{pred_start}\n")
                    f.write(f"predictions for the end of the song:\n{pred_end}\n")
            if l[1] < 10:
                current_model.save(current_path, f'{i+1}')
                voice_loader(current_path, data=np.array(pred_start), name="start")
                voice_loader(current_path, data=np.array(pred_end), name="end")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(e)
            continue
                   

def run_existing(name) -> None:
    """
    Runs a model saved on the disk, it will complete a prediction and show this. If sound files of this model do not exist they are created.

    Args:
        name (str): The name of the model to be run
    """    
    model = MusicModel()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load(os.path.join(dir_path, "models/"), name)

    # X = np.concatenate((one_hot_encode(0), [5]))
    # X = np.expand_dims(X, 0)
    # pred = model.predict(X, 100)
    
    sequence = load_sequence()
    pred = model.predict(sequence, 100)
    if not os.path.exists(os.path.join(music_path, name)):
        vl = VoiceLoader()
        start = model.predict(sequence[:10], 300)
        middle = model.predict(sequence[:300], 300)
        end = model.predict(sequence[:600], 300)
        vl(music_path, data=np.array(start), name=f"{name}-start")
        vl(music_path, data=np.array(middle), name=f"{name}-middle")
        vl(music_path, data=np.array(end), name=f"{name}-end")
    print(pred)


if __name__ == "__main__":
    # Start program: Either using argument train to train a new model with the provided name, run a saved model with the provided name or conduct a grid search saved under name
    args = sys.argv
    if len(args) < 2 or len(args) > 3:
        raise ValueError("Invalid command line, run: python main.py train|run|grid name")
    if args[1] == 'train':
        train_new(args[2])
    elif args[1] == 'run':
        run_existing(args[2])
    elif args[1] == 'grid':
        grid_search(args[2])
    else:
        raise ValueError("Invalid command line, run: python main.py train|run|grid name")
    
    
    
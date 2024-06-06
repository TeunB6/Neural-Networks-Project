import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer, Adam, SGD
from src.constants import INPUT_SIZE, OUTPUT_SIZE
from src.utils.device import fetch_device

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(0)

    def forward(self, x, h0 = None):
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(fetch_device()) if not h0 else h0 
        out, ht = self.rnn(x, h0)
        out = self.h2o(out[-1])  # Take the last output
        prob_out = self.softmax(out)
        # print(prob_out)
        return prob_out, ht

class MusicModel:  
    def __init__(self, loss_function = nn.CrossEntropyLoss(),
                       optimizer: Optimizer = Adam,
                       optimizer_args: dict = {"lr" : 0.0005},
                       epochs: int = 1, 
                       hidden_size: int = 256,
                       num_layers: int = 2,
                       verbose: int = 0,
                       ) -> None:
        # Set learning parameters
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.verbose = verbose
        self.loss = loss_function
        self.model = RNNModel(INPUT_SIZE, self.hidden_size, OUTPUT_SIZE, num_layers).to(fetch_device())
        self.optimizer : Optimizer = optimizer(self.model.parameters(), **optimizer_args)
    
    def fit(self, X, y) -> None:
        self.model.train()
        try:
            for e in range(self.epochs):
                if self.verbose > 0:
                    print(f"Starting Epoch {e+1}/{self.epochs}")
                
                loss_history = []
                data_len = len(X)
                for i, (data, target) in enumerate(zip(X, y)):
                    data_t = torch.tensor(data, dtype=torch.float32)
                    target_t = torch.tensor(target, dtype=torch.float32)
                    out, _ = self.model.forward(data_t)
                    loss = self.loss(out, target_t)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    loss_history.append(loss.item())
                    
                    if self.verbose > 1 and (i + 1) % 100 == 0:
                        print(out)
                        print(f"Iteration {i}/{data_len}: mean loss - {torch.mean(torch.tensor(loss_history[i + 1 - 100 : i]))}")
                
                if self.verbose > 0:
                    print(f"Epoch {e+1}/{self.epochs}: mean loss - {torch.mean(torch.tensor(loss_history))}")
        except Exception as e:
            print(f"Error occured: Saving Model...")
            self.save(name="error")
            raise e
                    
    
    def score(self, X, y) -> None:
        loss_history = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (data, target) in enumerate(zip(X, y)):
                out, _ = self.model.forward(data)
                loss = self.loss(out, target)                
                loss_history.append(loss.item())
            
            mean_loss = torch.mean(loss_history)
            if self.verbose > 0: 
                print(f"Scored {len(X)} data points: mean loss of {mean_loss}")
    
    @staticmethod
    def select_note(p:torch.Tensor):
        
        selected_note = np.random.choice(range(INPUT_SIZE), p=p.detach().numpy())
        one_hot = np.zeros((INPUT_SIZE))
        one_hot[selected_note] = 1
        return one_hot
        
        
    
    def predict(self, X, num_notes) -> list:
        note_vector = []    
        with torch.no_grad():
            out, ht = self.model.forward(X)
            data = self.select_note(out)
            note_vector.append(data)
            
            for _ in range(num_notes):
                out, ht = self.model.forward(data, ht)
                data = self.select_note(out)
                note_vector.append(data)
                
        return note_vector

    def save(self, file_path='./', name: str = "") -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self.model.state_dict(), file_path + f"music_model_{name}.pth")

    def load(self, file_path='./', name: str = "") -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self.model.load_state_dict(torch.load(file_path + f"music_model_{name}.pth"))

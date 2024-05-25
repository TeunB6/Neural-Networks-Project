import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from constants import INPUT_SIZE, OUTPUT_SIZE
from utils.device import fetch_device

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(fetch_device()) # idk if this makes sense bro cus ur init the hidden vector for every forward pass, this assumes x is a full timeseries vector
        out, _ = self.rnn(x, h0)
        out = self.h2o(out[:, -1, :])  # Take the last output
        return out

class MusicModel:
    
    def __init__(self, loss_function = nn.CrossEntropyLoss(),
                       optimizer: Optimizer = Adam,
                       optimizer_args: dict = {"lr" : 0.05, "weight_decay" : 0.01},
                       epochs: int = 3, 
                       hidden_size: int = 128,
                       num_layers: int = 1,
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
                for i, data, target in enumerate(zip(X, y)):
                    out = self.model.forward(data)
                    loss = self.loss(out, target)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    loss_history.append(loss.item())
                    
                    if self.verbose > 1 and (i + 1) % 100: 
                        print(f"Iteration {i}/{data_len}: mean loss - {torch.mean(loss_history[i + 1 - 100 : i])}")
                
                if self.verbose > 0:
                    print(f"Epoch {e+1}/{self.epochs}: mean loss - {torch.mean(loss_history)}")
        except Exception as e:
            print(f"Error occured: Saving Model...")
            self.save(name="error")
            raise e
                    
    
    def score(self, X, y) -> None:
        loss_history = []
        data_len = len(X)
        self.model.eval()
        
        with torch.no_grad():
            for i, data, target in enumerate(zip(X, y)):
                out = self.model.forward(data)
                loss = self.loss(out, target)                
                loss_history.append(loss.item())
            
            mean_loss = torch.mean(loss_history)
            if self.verbose > 0: 
                print(f"Scored {len(X)} data points: mean loss of {mean_loss}")
            

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

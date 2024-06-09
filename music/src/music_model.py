import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer, Adam, SGD
from src.constants import INPUT_SIZE, OUTPUT_SIZE, WEIGHT_VECTOR
from src.utils.device import fetch_device

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.device = fetch_device()
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.h2prob = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.h2num = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, x, h0 = None):
        h0 = torch.zeros(self.num_layers, self.hidden_size, device=self.device) if not h0 else h0
        out, ht = self.rnn(x, h0)
        out = out[-1]
        prob_out = self.h2prob(out)
        num_out = self.h2num(out)
        return prob_out.to(self.device), num_out.to(self.device), ht.to(self.device)

class MusicModel:  
    def __init__(self, loss_function = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_VECTOR)),
                       optimizer: Optimizer = Adam,
                       optimizer_args: dict = {"lr" : 0.001, "weight_decay" : 1},
                       epochs: int = 1, 
                       hidden_size: int = 128,
                       num_layers: int = 100,
                       verbose: int = 1,
                       ) -> None:
        # Set up 
        self.device = fetch_device()
        self.model = RNNModel(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layers).to(self.device)
        self.optimizer : Optimizer = optimizer(self.model.parameters(), **optimizer_args)
        # self.optimizer : Optimizer = optimizer(self.model.h2prob.parameters(), **optimizer_args)
        
        
        # Set learning parameters
        self.epochs = epochs
        self.verbose = verbose
        self.loss = loss_function.to(self.device)
        
    def fit(self, X, y) -> None:
        
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        try:
            for e in range(self.epochs):
                if self.verbose > 0:
                    print(f"Starting Epoch {e+1}/{self.epochs}")
                
                loss_history = []
                data_len = len(X)
                # X = torch.tensor(X, dtype=torch.float32, device=self.device)
                # y = torch.tensor(y, dtype=torch.float32, device=self.device)
                
                for i, (data, target) in enumerate(zip(X, y)):
                    target_note = target[:-1] if isinstance(target, torch.Tensor) else torch.tensor(target[:-1], dtype=torch.float32, device=self.device)
                    target_num = target[-1] if isinstance(target, torch.Tensor) else torch.tensor(target[:-1], dtype=torch.float32, device=self.device)
                    data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32, device=self.device)
                    
                    out_prob, out_num, _ = self.model.forward(data)
                    prob_loss = self.loss(out_prob, target_note)
                    num_loss = torch.nn.functional.mse_loss(out_num, target_num)
                    
                    total_loss = prob_loss # + num_loss
                    total_loss.backward()
                    self.optimizer.step()
                    
                    loss_history.append(total_loss.item())
                    
                    if self.verbose > 1 and (i + 1) % 100 == 0:
                        print(f"Iteration {i + 1}/{data_len}: mean loss - {np.mean(loss_history[i + 1 - 100 : i])}")
                        print(f"Latest Sample: target {torch.argmax(target_note), target_num},\n\t\t out {torch.argmax(out_prob), out_num}" + 
                              f" \n {out_prob, torch.sum(out_prob)}")
                
                if self.verbose > 0:
                    print(f"Epoch {e+1}/{self.epochs}: mean loss - {np.mean(loss_history)}")
        except Exception as e:
            print(f"Error occured: Saving Model...")
            self.save(name="error")
            raise e
                    
    
    def score(self, X, y) -> torch.Tensor:
        
        # X = torch.tensor(X).to(self.device)
        # y = torch.tensor(y).to(self.device)
        loss_history = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (data, target) in enumerate(zip(X, y)):
                target_note = target[:-1] if isinstance(target, torch.Tensor) else torch.tensor(target[:-1], dtype=torch.float32, device=self.device)
                target_num = target[-1] if isinstance(target, torch.Tensor) else torch.tensor(target[:-1], dtype=torch.float32, device=self.device)
                data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32, device=self.device)

                
                
                out, _ = self.model.forward(data)
                loss = self.loss(out, target_note)                
                loss_history.append(loss.item())
            
            mean_loss = torch.mean(loss_history)
            if self.verbose > 0: 
                print(f"Scored {len(X)} data points: mean loss of {mean_loss}")
        return mean_loss
    
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

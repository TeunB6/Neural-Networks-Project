import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import DataLoader
from src.constants import INPUT_SIZE, OUTPUT_SIZE, WEIGHT_VECTOR
from src.utils.device import fetch_device
from src.utils.preprocess import one_hot_decode
from src.utils.dataset import CustomDataset


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers, batch_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
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
        num_batches = x.shape[0]
        h0 = torch.zeros(self.num_layers, num_batches, self.hidden_size, device=self.device) if h0 is None else h0
        out, ht = self.rnn(x, h0)
        out = out[:, -1, :]
        prob_out = self.h2prob(out)
        num_out = self.h2num(out)
        return ht

class MusicModel:  
    def __init__(self, loss_function = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_VECTOR)),
                       optimizer: Optimizer = SGD,
                       optimizer_args: dict = {"lr" : 0.1, "momentum" : 0.9},
                       epochs: int = 10, 
                       hidden_size: int = 64,
                       num_layers: int = 2,
                       batch_size: int = 300,
                       verbose: int = 0,
                       ) -> None:
        # Set up 
        self.device = fetch_device()
        self.model = RNNModel(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layers, batch_size).to(self.device)
        self.optimizer : Optimizer = optimizer(self.model.parameters(), **optimizer_args)
        # self.optimizer : Optimizer = optimizer(self.model.h2prob.parameters(), **optimizer_args)
        
        # Set learning parameters
        self.epochs = epochs
        self.verbose = verbose
        self.loss = loss_function.to(self.device)
        self.batch_size = batch_size
    def sample(self, sequence):
        h0 = None
        data = []
        paired_sequence = [(sequence[i], sequence[i+1]) for i in range(len(sequence) - 1)]
        for note, next_note in paired_sequence:
            hidden = self.model.forward(note, h0)
            data.append((hidden, next_note))

        return data




    def fit(self, X, y) -> None:
        
        self.model.train()
        
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
        dataset = CustomDataset(X, y)

        
        try:
            for e in range(self.epochs):
                if self.verbose > 0:
                    print(f"Starting Epoch {e+1}/{self.epochs}")
                
                loss_history = []
                
                
                train_loader = DataLoader(dataset, self.batch_size, shuffle=True)
                data_len = len(train_loader)
                
                for i, (data, target) in enumerate(train_loader):
                    target_note = target[:, :-1] 
                    target_num = target[:, -1]
                    out_prob, out_num, _ = self.model.forward(data)
                    prob_loss = self.loss(out_prob, target_note)
                    # num_loss = torch.nn.functional.mse_loss(out_num, target_num)
                    
                    total_loss = prob_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    loss_history.append(total_loss.item())
                    
                    if self.verbose > 1:
                        print(f"Iteration {i + 1}/{data_len}: mean loss - {total_loss.item()}")
                        # print(f"Latest Sample: target {torch.argmax(target_note), target_num},\n\t\t out {torch.argmax(out_prob), out_num}" + 
                        #       f" \n {out_prob, torch.sum(out_prob)}")
                
                if self.verbose > 0:
                    print(f"Epoch {e+1}/{self.epochs}: mean loss - {np.mean(loss_history)}")
        except Exception as e:
            print(f"Error occured: Saving Model...")
            self.save(name="error")
            raise e
        
        # Plot loss curve
        if self.verbose > 2:
            plt.figure()
            plt.plot(loss_history)
            plt.show()

    
    def score(self, X, y) -> tuple[float, float]:
        
        loss_history = []
        acc_history = []
        self.model.eval()
        
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
        dataset = CustomDataset(X, y)
        
        test_loader = DataLoader(dataset, self.batch_size, shuffle=True)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                target_note = target[:, :-1] 
                target_num = target[:, -1]
                out_prob, out_num, _ = self.model.forward(data)
                
                # Calculate loss
                prob_loss = self.loss(out_prob, target_note)
                # num_loss = torch.nn.functional.mse_loss(out_num, target_num)
                
                total_loss = prob_loss                
                loss_history.append(total_loss.item())
                
                # Calculate Accuracy
                out_note = np.round(out_prob.cpu())
                total = out_prob.size(0)
                correct = total - (out_note != target_note.cpu()).sum().item() / 2                
                acc_history.append((correct / total))
                            
            mean_acc = np.mean(acc_history)
            mean_loss = np.mean(loss_history)
            if self.verbose > 0: 
                print(f"Scored {len(X)} data points: mean loss of {mean_loss}, mean accuracy of {mean_acc}")
        return mean_loss, mean_acc
    
    @staticmethod
    def select_note(p: torch.Tensor):
        probabilities = p.cpu().detach().numpy()[0]
        selected_note = np.random.choice(range(INPUT_SIZE - 1), p=probabilities)
        one_hot = np.zeros((INPUT_SIZE))
        one_hot[selected_note] = 1
        return one_hot
        
        
    
    def predict(self, X, num_notes) -> list:
        note_vector = []    
        X = torch.tensor([[X]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out_prob, out_num, ht = self.model.forward(X)
            data = self.select_note(out_prob)
            note_vector.append(one_hot_decode(data))
            
            for _ in range(num_notes):
                out_prob, out_num, ht = self.model.forward(torch.tensor([[data]], dtype=torch.float32, device=self.device), ht)
                data = self.select_note(out_prob)
                note_vector.append(one_hot_decode(data))
                print(out_prob)
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

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    @lr.setter
    def lr(self, new_lr):
        self.optimizer.param_groups[0]['lr'] = new_lr
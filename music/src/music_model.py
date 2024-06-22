import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import DataLoader
from src.constants import INPUT_SIZE, OUTPUT_SIZE, WEIGHT_VECTOR
from src.utils.device import fetch_device
from src.utils.preprocess import one_hot_decode, one_hot_max
from src.utils.dataset import CustomDataset
from typing import Optional

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.device = fetch_device()
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.init_weights(self.rnn)
        
    def init_weights(self, module: nn.Module):
        """Initializes the weights of the provided module in [-0.5,0.5], resulting in the network likely possesing the echo state property.

        Args:
            module (nn.Module): the module whose parameters should be initialized.
        """        
        for param in module.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.5, b=0.5)

    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward pass of the network

        Args:
            x (torch.Tensor): the (batched) time series to generate the hiddenstates from
            h0 (torch.Tensor, optional): The current hidden state of the network, when None the hidden state is initialized . Defaults to None.

        Returns:
           tuple[torch.Tensor, torch.Tensor] : A tuple of the final hidden vector relevant for the output and the full hidden state needed to continue the process for the next timestep
        """        
        with torch.no_grad():
            if h0 is None:
                h0 = (torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=self.device) if x.dim() > 2
                    else torch.zeros(self.num_layers, self.hidden_size, device=self.device))
            _, ht = self.rnn(x, h0)
        return ht[-1], ht

class MusicModel:  
    def __init__(self, prob_optimizer: Optimizer = SGD,
                       prob_optimizer_args: dict = {"lr" : 0.005, "momentum" : 0.9},
                       durr_optimizer: Optimizer = Adam,
                       durr_optimizer_args: dict = {"lr" : 0.001},
                       epochs: int = 30, 
                       hidden_size: int = 2048,
                       num_layers: int = 1,
                       batch_size: int = 1,
                       verbose: int = 0,
                       ) -> None:
        # Set up 
        
        self.device = fetch_device()
        self.reservoir = RNNModel(INPUT_SIZE, hidden_size, num_layers, batch_size).to(self.device)
        self.prob_model = nn.Sequential(nn.Linear(hidden_size, hidden_size //2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size // 2, OUTPUT_SIZE),
                                   nn.Sigmoid()).to(self.device)
                                   #nn.Softmax(dim=-1)).to(self.device)
        self.durr_model = nn.Sequential(nn.Linear(hidden_size, 1),
                                #    nn.ReLU(),
                                #    nn.Linear(hidden_size // 2, 1),
                                   nn.ReLU()).to(self.device)
        
        
        self.prob_optimizer : Optimizer = prob_optimizer(self.prob_model.parameters(), **prob_optimizer_args)
        self.durr_optimizer : Optimizer = durr_optimizer(self.durr_model.parameters(), **durr_optimizer_args)
        
        # Set learning parameters
        self.epochs = epochs
        self.verbose = verbose
        self.prob_loss_function =  nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_VECTOR)).to(self.device)
        self.durr_loss_function = nn.MSELoss().to(self.device)
        
        self.batch_size = batch_size
    
    
    def sample_reservoir(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Method for sampling the reservoir, this create a supervised learning dataset which can be used to train the FFN based on the hidden states of the recurrent reservoir.

        Args:
            sequence (np.ndarray): The time series to create the samples from

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing the data and corresponding labels sampled from the reservoir
        """        
        hidden = None
        data = []
        labels = []
        sequence = torch.tensor(np.array(sequence), dtype=torch.float32, device=self.device)
        paired_sequence = [(sequence[i], sequence[i+1]) for i in range(len(sequence) - 1)]
        for note, next_note in paired_sequence:
            reservoir_state, hidden = self.reservoir.forward(note.unsqueeze(0), hidden)
            data.append(reservoir_state.cpu())
            labels.append(next_note.cpu())

        return np.array(data), np.array(labels)


    def fit_ffn(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Method for training the feedforward network, utilises the two different loss functions to train the note and duration predictor networks. Based on the data sampled from the reservoir.

        Args:
            X (np.ndarray): Data from reservoir
            y (np.ndarray): Labels

        Raises:
            e: Any errors faced while running the model will be reraised after saving the current model
        """        
        # Put models in training mode
        self.prob_model.train()
        self.durr_model.train()
        
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
        dataset = CustomDataset(X, y)
        loss_history = []
        
        try:
            
            for e in range(self.epochs):
                if self.verbose > 0:
                    print(f"Starting Epoch {e+1}/{self.epochs}")
                
                epoch_loss = []
                
                
                train_loader = DataLoader(dataset, self.batch_size, shuffle=True)
                data_len = len(train_loader)
                
                for i, (data, target) in enumerate(train_loader):
                    # Get probability and duration target
                    target_prob = target[:, :-1] 
                    target_durr = target[:, -1].unsqueeze(-1)
                    
                    # Compute probability and duration predictions
                    out_prob = self.prob_model.forward(data)
                    out_durr = self.durr_model.forward(data)
                    
                    # Compute losses
                    prob_loss = self.prob_loss_function(out_prob, target_prob)
                    durr_loss = self.durr_loss_function(out_durr, target_durr)
                    
                    # Backwards passes
                    self.prob_optimizer.zero_grad()
                    self.durr_optimizer.zero_grad()
                    prob_loss.backward()
                    durr_loss.backward()
                    self.prob_optimizer.step()
                    self.durr_optimizer.step()
                    
                    epoch_loss.append((prob_loss.item(), durr_loss.item()))
                    
                    if self.verbose > 1:
                        print(f"Iteration {i + 1}/{data_len}:  loss prob - {prob_loss.item()}, loss durr - {durr_loss.item()}")
                
                mean_loss = np.mean(epoch_loss, axis=0)
                loss_history.append(mean_loss)
                
                if self.verbose > 0:
                    print(f"Epoch {e+1}/{self.epochs}: mean loss prob - {mean_loss[0]}, mean loss durr - {mean_loss[1]}")
        except Exception as e:
            print(f"Error occured: Saving Model...")
            self.save(name="error")
            raise e
        
        # Plot loss curve
        if self.verbose > 0:
            plt.figure()
            plot_data = np.array(loss_history)
            plt.subplot(121)
            plt.plot(plot_data[:, 0])
            plt.subplot(122)
            plt.plot(plot_data[:, 1])
            plt.show()

    
    def score_ffn(self, X:np.ndarray, y:np.ndarray) -> tuple[float, float]:
        """
        Evaluates the FFN based on the testing datasets provided.

        Args:
            X (np.ndarray): Data
            y (np.ndarray): Labels

        Returns:
            tuple[float, float]: Tuple of mean loss and accuracy across the testing set.
        """        
        loss_history = []
        acc_history = []
        
        # Put models in evaluation mode
        self.prob_model.eval()
        self.durr_model.eval()
        

        
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
        dataset = CustomDataset(X, y)
        
        test_loader = DataLoader(dataset, self.batch_size, shuffle=True)
        
        with torch.no_grad():
            for data, target in test_loader:
                # Get probability and duration target
                target_prob = target[:, :-1] 
                target_durr = target[:, -1].unsqueeze(-1)
                
                # Compute probability and duration predictions
                out_prob = self.prob_model.forward(data)
                out_durr = self.durr_model.forward(data)
                
                # Calculate loss
                prob_loss = self.prob_loss_function(out_prob, target_prob)
                durr_loss = self.durr_loss_function(out_durr, target_durr)
                total_loss = prob_loss + durr_loss            
                loss_history.append(total_loss.item())
                
                # Calculate Accuracy
                out_note = [one_hot_max(arr) for arr in out_prob.cpu().numpy()]
                correct = sum(one_hot_decode(pred) == one_hot_decode(true) for pred, true in zip(out_note, target_prob.cpu()) if sum(true) != 0)
                total = len(target)
                acc_history.append(correct / total)

                            
            mean_acc = np.mean(acc_history)
            mean_loss = np.mean(loss_history)
            if self.verbose > 0: 
                print(f"Scored {len(X)} data points: mean loss of {mean_loss}, mean accuracy of {mean_acc}")
        return mean_loss, mean_acc
    
    @staticmethod
    def select_note(p: torch.Tensor) -> torch.Tensor:
        """
        Staticemthod for selecting the note based on the model output, if the FFN output is a probability distribution then these are used to randomize the note, otherwise the maximum value is chosen.

        Args:
            p (torch.Tensor): tensor of either probabilities or values taken from the network

        Returns:
            torch.Tensor: The one hot encoding from the selected note
        """        
        probabilities = p.cpu().detach().numpy()
        if sum(probabilities) == 1:
            selected_note = np.random.choice(range(INPUT_SIZE - 1), p=probabilities)
        else:
            selected_note = np.argmax(probabilities)
        one_hot = torch.zeros((INPUT_SIZE))
        one_hot[selected_note] = 1
        return one_hot
        
        
    
    def predict(self, X: np.ndarray | list[np.ndarray], num_notes: int) -> list:
        """Generates a prediction of the continuation of sequence X, the output is the raw time series as can be interpreted by the voice loader class

        Args:
            X (np.ndarray | list[np.ndarray]): The starting sequence, either a np.ndarray or a list of np.ndarrays
            num_notes (int): the amount of different notes to predict into the future (model time steps)

        Returns:
            list: the resulting time series
        """        
        output_list = []    
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            sample, ht = self.reservoir.forward(X)
            out_prob = self.prob_model.forward(sample)
            out_durr = self.durr_model.forward(sample)
            selected_note = self.select_note(out_prob)
            output_list += [one_hot_decode(selected_note)] * int(out_durr + 1)
            
            for _ in range(num_notes):
                selected_note.unsqueeze_(0)
                selected_note = selected_note.to(self.device)
                sample, ht = self.reservoir.forward(selected_note, ht)
                
                out_prob = self.prob_model.forward(sample)
                out_durr = self.durr_model.forward(sample)
                
                selected_note = self.select_note(out_prob)
                output_list += [one_hot_decode(selected_note)] * int(out_durr + 1)

        return output_list

    def save(self, file_path='./', name: str = "") -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self.prob_model.state_dict(), file_path + f"{name}_FFN_prob.pth")
        torch.save(self.durr_model.state_dict(), file_path + f"{name}_FFN_durr.pth")
        torch.save(self.reservoir.state_dict(), file_path + f"{name}_reservoir.pth")

    def load(self, file_path='./', name: str = "") -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self.prob_model.load_state_dict(torch.load(file_path + f"{name}_FFN_prob.pth"))
        self.durr_model.load_state_dict(torch.load(file_path + f"{name}_FFN_durr.pth"))
        self.reservoir.load_state_dict(torch.load(file_path + f"{name}_reservoir.pth"))
        
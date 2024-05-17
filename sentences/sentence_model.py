import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
all_letters = "abcdefghijklmnopqrstuvwxyz. "
num_letters = len(all_letters)

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = F.tanh(self.i2h(x) + self.h2h(h))
        o = self.h2o(h)
        o = self.softmax(o)
        return o, h


class SentenceModel:
    def __init__(self, hidden_size: int = 128,
                 lr: float = 0.005
                 ) -> None:
        self.model = RecurrentNeuralNetwork(num_letters, hidden_size, num_letters)
        self.hidden_size = hidden_size
        self.lr = lr
        self.loss_function = nn.NLLLoss()
        self.init_hidden()
        
    @staticmethod
    def char_to_tensor_out(char: str):
        if type(char) != str:
            raise TypeError(f"confusion: char {char} is not a str but {type(char)}")
        if len(char) != 1:
            raise(ValueError(f"attempted to encode on too long char string: {char}"))
        
        char_index = all_letters.find(char)
        return torch.tensor([char_index])

    @staticmethod
    def char_to_tensor_in(char: str):
        tensor = torch.zeros(1, num_letters)
        char_index = all_letters.find(char)
        tensor[0][char_index] = 1
        return tensor
    
    @staticmethod
    def string_to_tensor(string: str):
        tensor = torch.zeros(len(string), 1, num_letters)
        
        for li, char in enumerate(string):
            tensor[li] = SentenceModel.char_to_tensor_in(char)
        
        return tensor

    def init_hidden(self):
        self.hidden = torch.zeros(1, self.hidden_size)
    
    def predict_sentence(self, string: str) -> str:    
        self.init_hidden()
        with torch.no_grad():
            for char in string[:-1]:
                char_t = self.char_to_tensor_in(char)
                _, self.hidden = self.model.forward(char_t, self.hidden)
            
            while True:
                char_t = self.char_to_tensor_in(string[-1])
                o, self.hidden = self.model.forward(char_t, self.hidden)
                next_char_idx = torch.argmax(o)
                next_char = all_letters[next_char_idx]
                string += next_char
                if next_char == '.':
                    return string
    
    def predict_char(self, string: str) -> str:
        self.init_hidden()
        with torch.no_grad():
            for char in string:
                char_t = self.char_to_tensor_in(char)
                o, self.hidden = self.model.forward(char_t, self.hidden)
            next_char = all_letters[torch.argmax(o)]
        return next_char
    
    def _train_sample(self, string_tensor: torch.Tensor, next_char_tensor: torch.Tensor):
        self.init_hidden()
        
        self.model.zero_grad()
        
        for t in string_tensor:
            output, self.hidden = self.model.forward(t, self.hidden)
        loss = self.loss_function(output, next_char_tensor)
        loss.backward()
        
        # update rule
        for p in self.model.parameters():
            p.data.add_(p.grad.data, alpha=-self.lr)
        
        return output, loss.item()
        
    def train(self, strings: list[str], target: list[str], shuffle: bool = False):
        
        self.model.train()
        
        data = list(zip(strings, target))
        if shuffle:
            random.shuffle(data)
        loss_history = []
        try:
            for iter, (string, next_char) in enumerate(data):
                string_tensor = self.string_to_tensor(string)
                char_tensor = self.char_to_tensor_out(next_char)                
                o, loss = self._train_sample(string_tensor, char_tensor)
                
                if iter % 100 == 0:
                    print(f"{iter}/{len(data)} loss = {loss}") 
                    loss_history.append(loss)
        except Exception as e:
            print("Exception occured while training, saving model...")
            self.save(name=f'latestloss{loss_history[-1]}')
            raise e            
            
        plt.figure()
        plt.plot(loss_history)
        plt.show()
        self.model.eval()
    
    def save(self, file_path='./', name: str = "") -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self.model.state_dict(), file_path + f"sentence_model_{name}.pth")

    def load(self, file_path='./', name: str = "") -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self.model.load_state_dict(torch.load(file_path + f"sentence_model_{name}.pth"))

        
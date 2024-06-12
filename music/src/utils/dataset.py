from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def __getitem__(self, index) -> tuple:
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


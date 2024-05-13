import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o

class TicTacToeModel(nn.Module):
    def __init__(self, lr : int) -> None:
        super(TicTacToeModel, self).__init__()

        self.fc1 = nn.Linear(27, 54)
        self.fc2 = nn.Linear(54, 9)

        self.loss = nn.MSELoss()
        
        self.optimizer = o.SGD(self.parameters(), lr = lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
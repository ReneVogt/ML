import torch.nn as nn
import torch.nn.functional as F

class Connect4Cnn(nn.Module):
    def __init__(self):
        super(Connect4Cnn, self).__init__()

        self.conv = nn.Conv2d(1, 64,  kernel_size=3, stride=1, padding=0)
        self.linear = nn.Linear(256, 128)
        self.out = nn.Linear(128, 7)

    def forward(self, x):
        self.convout = x = F.relu(self.conv(x))
        self.poolout = x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.flatten(start_dim=1)
        self.linearout = x = F.relu(self.linear(x))
        self.finalout = x = self.out(x)
        return x
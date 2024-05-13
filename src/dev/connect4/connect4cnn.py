import torch.nn as nn
import torch.nn.functional as F

class Connect4Cnn(nn.Module):
    def __init__(self):
        super(Connect4Cnn, self).__init__()

        # input: [B,3,6,7]
        self.conv = nn.Conv2d(3, 64,  kernel_size=4, stride=1, padding=1) # [B, 64, (6 + 2*1 - 4)/1 + 1 = 5, (7 + 2*1 - 4)/1 + 1 = 6]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # [B, 64, (5 - 2)/2 + 1 = 2, (6 - 2)/2 + 1 = 3] 
        self.flatten = nn.Flatten(start_dim=1) # [B, 64*2*3 = 384]
        self.linear = nn.Linear(384, 128)
        self.out = nn.Linear(128, 7)        

    def forward(self, x):
        self.convout = x = F.relu(self.conv(x))
        self.poolout = x = self.pool(x)
        x = self.flatten(x)
        self.linearout = x = F.relu(self.linear(x))
        self.finalout = x = self.out(x)
        return x
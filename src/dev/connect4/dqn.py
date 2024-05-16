import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o

class Connect4Dqn(nn.Module):
    def __init__(self, lr : float):
        super(Connect4Dqn, self).__init__()

        # input: [B,3,6,7]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # [B, 16, 6, 7]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # [B, 32, 6, 7]

        self.flatten = nn.Flatten(start_dim=1) # [B, 1344]

        self.fc1 = nn.Linear(1344, 672)
        self.fc2 = nn.Linear(672, 96)
        self.fc3 = nn.Linear(96, 7)

        self.loss = nn.MSELoss()
        self.optimizer = o.SGD(self.parameters(), lr)

    def forward(self, x):
        self.conv1out = x = F.relu(self.conv1(x))
        self.conv2out = x = F.relu(self.conv2(x))

        x = self.flatten(x)

        self.fc1out = x = F.relu(self.fc1(x))
        self.fc2out = x = F.relu(self.fc2(x))
        self.out = x = self.fc3(x)
        return x
import torch.nn as nn
import torch.nn.functional as F

class Connect4Cnn(nn.Module):
    def __init__(self):
        super(Connect4Cnn, self).__init__()

        self.preoutsize = 128*5*5

        self.conv1 = nn.Conv2d(1, 64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.out = nn.Linear(self.preoutsize, 7)

    def forward(self, x):
        self.conv1out = x = F.relu(self.conv1(x))
        self.conv2out = x = F.relu(self.conv2(x))
        self.pool2out = x = self.pool2(x)
        self.conv3out = x = F.relu(self.conv3(x))
        self.pool3out = x = self.pool3(x)
        x = x.view(-1, self.preoutsize)
        x = self.out(x)
        return x
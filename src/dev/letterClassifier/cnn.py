import torch as T
import torch.nn as nn
import torch.nn.functional as F

class LetterClassifierCnn(nn.Module):
    def __init__(self):
        super(LetterClassifierCnn, self).__init__()

        # input: [B,1,28,28]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # [B,32,26,26]
        # pool with k=2, s=2, p=0                       # [B,32,13,13]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # [B,64,11,11]
        # adaptive pooling                              # [B,64,5,5]
        self.flatten = nn.Flatten(start_dim=1)          # [B,1600]

        self.fc1 = nn.Linear(1600, 128)                 # [B,128]
        self.fc2 = nn.Linear(128, 26)                   # [B,26]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool2d(x, 5)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

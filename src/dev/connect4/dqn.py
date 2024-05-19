import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o

def exportOnnx(model : nn.Module, fileName : str) -> None:
    dummy_input = T.zeros([1,3,6,7])
    T.onnx.export(model, dummy_input, f"{fileName}.onnx");


# class Connect4DqnOld(nn.Module):
#     def __init__(self, lr : float):
#         super(Connect4DqnOld, self).__init__()

#         # input: [B,3,6,7]
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # [B, 16, 6, 7]
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # [B, 32, 6, 7]

#         self.flatten = nn.Flatten(start_dim=1) # [B, 1344]

#         self.fc1 = nn.Linear(1344, 672)
#         self.fc2 = nn.Linear(672, 96)
#         self.fc3 = nn.Linear(96, 7)

#         self.loss = nn.MSELoss()
#         self.optimizer = o.SGD(self.parameters(), lr)

#     def forward(self, x):
#         self.conv1out = x = F.relu(self.conv1(x))
#         self.conv2out = x = F.relu(self.conv2(x))

#         x = self.flatten(x)

#         self.fc1out = x = F.relu(self.fc1(x))
#         self.fc2out = x = F.relu(self.fc2(x))
#         self.out = x = self.fc3(x)
#         return x
    
# class Connect4Dqn(nn.Module):
#     def __init__(self, lr : float):
#         super(Connect4Dqn, self).__init__()

#         # input: [B,3,6,7]

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1)   # [B,16,7,8]
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # [B,32,7,8]
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # [B,32,5,6]

#         self.flatten = nn.Flatten(start_dim=1) # [B, 960]

#         self.fc1 = nn.Linear(960, 480)
#         self.fc2 = nn.Linear(480, 120)
#         self.fc3 = nn.Linear(120, 7)

#         self.loss = nn.MSELoss()
#         self.optimizer = o.SGD(self.parameters(), lr)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))

#         x = self.flatten(x)

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
class Connect4Dqn(nn.Module):
    def __init__(self, lr : float):
        super(Connect4Dqn, self).__init__()

        # input: [B,3,6,7]

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)    # [B,16,6,7]
        self.flatten = nn.Flatten(start_dim=1) # [B, 672]

        self.fc1 = nn.Linear(672, 336)
        self.fc2 = nn.Linear(336, 84)
        self.fc3 = nn.Linear(84, 7)

        self.loss = nn.MSELoss()
        self.optimizer = o.SGD(self.parameters(), lr)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
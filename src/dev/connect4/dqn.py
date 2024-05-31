import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o

def exportOnnx(model : nn.Module, fileName : str) -> None:
    dummy_input = T.zeros([1,3,6,7])
    T.onnx.export(model, dummy_input, f"{fileName}.onnx");

class Connect4Dqn(nn.Module):
    def __init__(self, lr : float):
        super(Connect4Dqn, self).__init__()

        # input: [B,3,6,7]

        self.conv = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=1)    # [B,128,5,6]
        self.flatten = nn.Flatten(start_dim=1) # [B, 3840]

        self.fc1 = nn.Linear(3840, 1920)
        self.fc2 = nn.Linear(1920, 640)
        self.fc3 = nn.Linear(640, 7)

        self.loss = nn.MSELoss()
        self.optimizer = o.SGD(self.parameters(), lr)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
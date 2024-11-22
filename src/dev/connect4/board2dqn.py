import torch as T
import torch.nn as nn
import torch.nn.functional as F
from connect4.board import Connect4Board

@T.no_grad()
def createStateTensor(env : Connect4Board) -> T.Tensor:
    state = T.zeros([3,6,7], requires_grad=False)
    for column in range(7):
        height = env._getColumnHeight(column)
        state[0,height:,column] = 1
        for row in range(height):
            player = 1 if env[column, row] == env.Player else 2
            state[player,row,column] = 1
    return state

@T.no_grad()
def getBestAction(model : nn.Module, env : Connect4Board ) -> int:
    training = model.training
    model.eval()
    qs = model.forward(createStateTensor(env).unsqueeze(0))[0]
    action = max([a for a in env.ValidMoves], key = lambda x: qs[x]) 
    if training:
        model.train()
    return action


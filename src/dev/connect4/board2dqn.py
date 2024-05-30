import torch as T
import torch.nn as nn
from connect4.board import Connect4Board

_NEGINF = float('-inf')

def createStateTensor(env : Connect4Board) -> T.Tensor:
    state = T.zeros(3, 6, 7)
    for column in range(7):
        for row in range(6):
            player = env[column, row]
            if player > 0 and env.Player == Connect4Board.PLAYER2:
                player = 3 - player
            state[player, row, column] = 1
    return state.float()


@T.no_grad()
def getBestAction(model : nn.Module, state : T.Tensor, validMoves : list[int] ) -> int:
    training = model.training
    model.eval()
    qs = model.forward(state.unsqueeze(0))[0]
    action = max([a for a in validMoves], key = lambda x: qs[x])
    if training:
        model.train()
    return action


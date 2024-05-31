import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from board import Connect4Board
from board2dqn import getBestAction, createStateTensor

@T.no_grad()
def getValidationOpponentMove(model : nn.Module, state : T.Tensor, validMoves : list[int], omega : float) -> int:
    if len(validMoves) == 1:
        return validMoves[0]
    
    if np.random.uniform(0,1) < omega:
        return np.random.choice(validMoves)

    qvalues = model(state.unsqueeze(0)).squeeze()
    validqs = T.tensor([qvalues[a] for a in validMoves])
    probs = F.softmax(validqs, dim=0)
    action = validMoves[T.multinomial(probs, num_samples=1)]

    return action

@T.no_grad()
def _playValidationGame(model : nn.Module, qplayer : int, omega : float) -> Connect4Board:
    env = Connect4Board()
    while not env.Finished:
        state = createStateTensor(env)
        validMoves = [a for a in range(7) if env.is_valid(a)]
        action = getBestAction(model, state, validMoves) if qplayer == env.Player else getValidationOpponentMove(model, state, validMoves, omega)
        env.move(action)
    return env

@T.no_grad()
def validate(model : nn.Module, gamesPerPlayer : int, omega : float) -> None:
    print(f'Validation with {100*(omega)}% random moves for {gamesPerPlayer} games per player.')

    training = model.training
    model.eval()

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(model, Connect4Board.PLAYER1, omega)
        gk = board.gameKey
        games.add(gk)
        if board.Winner == Connect4Board.PLAYER1:
            wins += 1
        elif board.Winner == Connect4Board.PLAYER2:
            losses += 1
        else:
            draws += 1

    print(f'Player 1: {wins} won, {losses} lost, {draws} draws -> {100*wins/gamesPerPlayer:.2f}%, div: {100*len(games)/gamesPerPlayer:.2f}%')

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(model, Connect4Board.PLAYER2, omega)
        gk = board.gameKey
        games.add(gk)
        if board.Winner == Connect4Board.PLAYER2:
            wins += 1
        elif board.Winner == Connect4Board.PLAYER1:
            losses += 1
        else:
            draws += 1

    print(f'Player 2: {wins} won, {losses} lost, {draws} draws -> {100*wins/gamesPerPlayer:.2f}%, div: {100*len(games)/gamesPerPlayer:.2f}%')

    if training:
        model.train()

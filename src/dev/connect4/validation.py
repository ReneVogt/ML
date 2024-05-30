import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from board import Connect4Board
from board2dqn import getBestAction, createStateTensor

@T.no_grad()
def getValidationOpponentMove(model : nn.Module, env : Connect4Board, omega : float) -> int:
    validMoves = [action for action in range(7) if env.is_valid(action)]
    if len(validMoves) == 1:
        return validMoves[0]
    
    if np.random.uniform(0,1) < omega:
        return np.random.choice(validMoves)

    training = model.training
    model.eval()
    state = createStateTensor(env)
    qvalues = model(state.unsqueeze(0)).squeeze()
    validqs = T.tensor([qvalues[a] for a in validMoves])
    probs = F.softmax(validqs, dim=0)
    action = validMoves[T.multinomial(probs, num_samples=1)]

    if training:
        model.train()

    return action

@T.no_grad()
def _playValidationGame(model : nn.Module, qplayer : int, omega : float) -> Connect4Board:
    env = Connect4Board()
    while not env.Finished:
        action = getBestAction(model, env) if qplayer == env.Player else getValidationOpponentMove(model, env, omega)
        env.move(action)
    return env

@T.no_grad()
def validate(model : nn.Module, gamesPerPlayer : int, omega : float) -> None:
    print(f'Validation with {100*(omega)}% random moves for {gamesPerPlayer} games per player.')

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

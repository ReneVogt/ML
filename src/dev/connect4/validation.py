import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from board import Connect4Board
from board2dqn import getBestAction, createStateTensor
from mcts import Connect4MCTS

@T.no_grad()
def _playValidationGame(model : nn.Module, qplayer : int, strength : int) -> Connect4Board:
    env = Connect4Board()
    mcts = Connect4MCTS(strength)
    while not env.Finished:
        if qplayer == env.Player:
            state = createStateTensor(env)
            validMoves = [a for a in range(7) if env.is_valid(a)]
            action = getBestAction(model, state, validMoves) 
        else:
            action = mcts.getMove(env)
        env.move(action)
    return env

@T.no_grad()
def validate(model : nn.Module, gamesPerPlayer : int, strength : int) -> None:
    print(f'Validation with MCTS at {strength} games for {gamesPerPlayer} games per player.')

    training = model.training
    model.eval()

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(model, Connect4Board.PLAYER1, strength)
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
        board = _playValidationGame(model, Connect4Board.PLAYER2, strength)
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

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import copy
from board import Connect4Board
from board2dqn import getBestAction, createStateTensor
from mcts import Connect4MCTS


@T.no_grad()
def _playValidationGame(model : nn.Module, qplayer : int, mctsGames : int) -> Connect4Board:
    env = Connect4Board()
    mcts = Connect4MCTS(mctsGames)
    while not env.Finished:
        if qplayer == env.Player:
            action = getBestAction(model, env) 
        else:
            action = mcts.getMove(env)
        env.move(action)
    return env

def _validationProcess(model : nn.Module, qplayer : int, games : int, mctsGames : int) -> tuple[int, int, int, int]:
    wins = draws = losses = 0
    gameKeys = set()
    for _ in range(games):
        board = _playValidationGame(model, qplayer, mctsGames)
        gameKeys.add(board.gameKey)
        if board.Winner == qplayer:
            wins += 1
        elif board.Winner != Connect4Board.EMPTY:
            losses += 1
        else:
            draws += 1

    return (qplayer, wins, draws, losses, gameKeys)

@T.no_grad()
def validate(model : nn.Module, gamesPerPlayer : int, processesPerPlayer : int, mctsGames : int) -> None:
    if processesPerPlayer < 1:
        raise ValueError('At least one validation process per player is required.')
    if gamesPerPlayer < 1:
        raise ValueError('At least one validation game per player is required.')
    if mctsGames < 1:
        raise ValueError('At least one mcts game per move is required.')
    
    print(f'Validation with {gamesPerPlayer} games per player on {processesPerPlayer} processes each, MCTS with {mctsGames} games.')

    training = model.training
    model.eval()

    gamesPerProcess = gamesPerPlayer // processesPerPlayer
    arguments = [(copy.deepcopy(model), qplayer, gamesPerProcess, mctsGames) for _ in range(processesPerPlayer) for qplayer in [Connect4Board.PLAYER1, Connect4Board.PLAYER2]]
    with mp.Pool(processes=len(arguments)) as pool:
        results = pool.starmap(_validationProcess, arguments)

    if training:
        model.train()

    wins1 = losses1 = draws1 = wins2 = losses2 = draws2 = 0
    games1 = set()
    games2 = set()

    for (qplayer, wins, draws, losses, games) in results:
        if qplayer == Connect4Board.PLAYER1:
            games1 = games1.union(games)
            wins1 += wins
            losses1 += losses
            draws1 += draws
        else:
            games2 = games2.union(games)
            wins2 += wins
            losses2 += losses
            draws2 += draws

    total1 = wins1+losses1+draws1
    unique1 = len(games1)
    total2 = wins2+losses2+draws2
    unique2 = len(games2)

    print(f'Player 1: {wins1} won, {losses1} lost, {draws1} draws -> {100*wins1/total1:.2f}%, div: {100*unique1/total1:.2f}%')
    print(f'Player 2: {wins2} won, {losses2} lost, {draws2} draws -> {100*wins2/total2:.2f}%, div: {100*unique2/total2:.2f}%')


import torch as T
from board import Connect4Board
from agent import Connect4Agent, createStateTensor

@T.no_grad()
def _playValidationGame(agent : Connect4Agent, qplayer : int, omega : float) -> Connect4Board:
    board = Connect4Board()
    while not board.Finished:
        state = createStateTensor(board)
        action = agent.getBestAction(state, board.ValidMovesMask) if qplayer == board.Player else agent.getValidationOpponentMove(state, board.ValidMovesMask, omega)
        board.move(action)
    return board

@T.no_grad()
def validate(agent : Connect4Agent, gamesPerPlayer : int, omega : float) -> None:
    print(f'Validation with {100*(omega)}% random moves')

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(agent, Connect4Board.PLAYER1, omega)
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
        board = _playValidationGame(agent, Connect4Board.PLAYER2, omega)
        gk = board.gameKey
        games.add(gk)
        if board.Winner == Connect4Board.PLAYER2:
            wins += 1
        elif board.Winner == Connect4Board.PLAYER1:
            losses += 1
        else:
            draws += 1

    print(f'Player 2: {wins} won, {losses} lost, {draws} draws -> {100*wins/gamesPerPlayer:.2f}%, div: {100*len(games)/gamesPerPlayer:.2f}%')

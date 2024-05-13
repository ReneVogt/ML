import sys
sys.path.append('../')

import torch.nn as nn
from agent import TicTacToeAgent
from tictactoe import TicTacToeBoard


#
# TicTacToe validation 
#
def _validateTicTacToeGame(agent : TicTacToeAgent, env : TicTacToeBoard, results):
    """
    Recursively plays a single TicTacToe game by choosing a
    random move followed by a max-Q move.
    """
    state, player, opponent = env.state, env.player, env.opponent
    for action in [a for a in range(9) if env.is_valid(a)]:
        env.move(action)
        if env.is_won():
            results['losses'] += 1
        elif env.is_full():
            results['draws'] += 1
        else:
            q = agent.getBestAction(env.stateTensor, env.validMovesMask)
            env.move(q)
            if env.is_won():
                results['wins'] += 1
            elif env.is_full():
                results['draws'] += 1
            else:
                _validateTicTacToeGame(agent, env, results)
        env.board, env.player, env.opponent = list(state), player, opponent

def validateTicTacToe(agent : TicTacToeAgent):
    """
    Validates a TicTacToe agent by letting it play as X against all possible move
    and then as O against all possble moves and counting the respective wins, draws
    and losses.
    """
    train = agent.training
    agent.eval()

    try:
        env = TicTacToeBoard()
        q = agent.getBestAction(env.stateTensor, env.validMovesMask)
        env.move(q)
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        _validateTicTacToeGame(agent, env, results)
        nonloss = results['wins'] + results['draws']
        total = results['losses'] + nonloss
        print(f"Cross: {100*nonloss/total:.2f}% of {total} ({results['wins']}/{results['draws']}/{results['losses']})")

        env = TicTacToeBoard()
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        _validateTicTacToeGame(agent, env, results)
        nonloss = results['wins'] + results['draws']
        total = results['losses'] + nonloss
        print(f"Circle: {100*nonloss/total:.2f}% of {total} ({results['wins']}/{results['draws']}/{results['losses']})")

    finally:
        if train:
            agent.train()
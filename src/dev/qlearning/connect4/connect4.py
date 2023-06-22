import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Connect4:   
    def __init__(self):
        self.board = [[], [], [], [], [], [], []]
        self.player = 1
        self.opponent = 2
        self.winner = 0
        self.full = False
        self.history = []        

    def is_valid(self, action):
        return action >= 0 and action < 7 and len(self.board[action]) < 6

    def move(self, action):
        if not self.is_valid(action):
            raise ValueError(f'Column {action} is already filled.')
        if self.winner != 0:
            raise ValueError(f'Game is already finished.')

        self.board[action].append(self.player)

        row = len(self.board[action]) - 1
        pattern = ''.join(['X' if self.player == 1 else 'O'] * 4)

        # check horizontal win
        if pattern in ''.join([' ' if len(self.board[col]) <= row else 'X' if self.board[col][row] == 1 else 'O' for col in range(7)]):
            self.winner = self.player
        # check vertical win
        elif pattern in ''.join(['X' if self.board[action][r] == 1 else 'O' for r in range(len(self.board[action]))]):
            self.winner = self.player
        else:
            # check diagonal win ll to ur
            indices = [(action + (x - 4), row + (x - 4)) for x in range(7)]
            diagonal = ''.join([' ' if col < 0 or col > 6 or row < 0 or row >= len(self.board[col]) else 'X' if self.board[col][row] == 1 else 'O' for col, row in indices])
            if pattern in diagonal:
                self.winner = self.player
            else:
                # check diagonal win ul to lr
                indices = [(action + (x - 3), row - (x - 3)) for x in range(7)]
                diagonal = ''.join([' ' if col < 0 or col > 6 or row < 0 or row >= len(self.board[col]) else 'X' if self.board[col][row] == 1 else 'O' for col, row in indices])
                if pattern in diagonal:
                    self.winner = self.player

        self.player, self.opponent = self.opponent, self.player
        self.history.append(action)
        self.full = all(len(self.board[i]) == 6 for i in range(7))

    def undo(self):
        if len(self.history) == 0:
            raise ValueError("The board is already empty.")
        
        action = self.history.pop()
        self.board[action].pop()
        self.player, self.opponent = self.opponent, self.player
        self.winner = 0
        self.full = False

    @property
    def state(self):
        return F.one_hot(torch.tensor([self.board[col][row] if len(self.board[col]) > row else 0 for col in range(7) for row in reversed(range(6))]), 3).view(-1).float()

    def render(self):
        for row in reversed(range(6)):
            signs = [self.board[col][row] if len(self.board[col]) > row else 0 for col in range(7)]
            signs = [' ' if x == 0 else 'X' if x == 1 else 'O' for x in signs]
            print(f"|{''.join(signs)}|")


@torch.no_grad()
def validate_full(model : nn.Module, loginterval = None):
    train = model.training
    model.eval()
    
    def increasegames(results):
        results['games'] += 1
        if not loginterval is None and  results['games'] % loginterval == 0:
            wins = results['wins']
            losses = results['losses']
            draws = results['draws']
            print(f"full validation: {results['games']} games, {100*(wins+draws)/(wins+draws+losses):.2f}%.")

    def validategame(model : nn.Module, env : Connect4, results):
        for action in [a for a in range(7) if env.is_valid(a)]:
            env.move(action)
            if env.winner != 0:
                results['losses'] += 1
                increasegames(results)
            elif env.full:
                results['draws'] += 1
                increasegames(results)
            else:
                q = model(env.state)
                qa = max([a for a in range(7) if env.is_valid(a)], key = lambda x: q[x])
                env.move(qa)
                if env.winner != 0:
                    results['wins'] += 1
                    increasegames(results)
                elif env.full:
                    results['draws'] += 1
                    increasegames(results)
                else:
                    validategame(model, env, results)
                env.undo()

            env.undo()

    env = Connect4()
    q = model(env.state)
    qa = max([a for a in range(9) if env.is_valid(a)], key = lambda x: q[x])
    env.move(qa)
    crossResults = {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    validategame(model, env, crossResults)
    env = Connect4()
    circleResults = {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    validategame(model, env, circleResults)
    
    if train:
        model.train()

    return crossResults, circleResults

@torch.no_grad()
def validate(model : nn.Module, games):
    train = model.training
    model.eval()

    wins = 0
    draws = 0
    losses = 0

    for _ in range(games):
        done = False
        qplayer = random.choice([1, 2])
        env = Connect4()
        while not done:
            if qplayer == env.player:
                q = model(env.state)
                action = max([a for a in range(7) if env.is_valid(a)], key = lambda x: q[x])
            else:
                action = random.choice([a for a in range(7) if env.is_valid(a)])

            env.move(action)

            if env.winner != 0:
                if env.winner == qplayer:
                    wins += 1
                else:
                    losses += 1
                done = True
            elif env.full:
                draws += 1
                done = True

    if train:
        model.train()

    return wins, draws, losses
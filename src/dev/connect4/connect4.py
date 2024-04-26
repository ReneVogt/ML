import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Connect4:   
    def __init__(self):
        self.board = torch.zeros(3, 6, 7, dtype = torch.float32)
        self.board[0, :, :] = 1
        self.heights = [0, 0, 0, 0, 0, 0, 0]
        self.player = 1
        self.opponent = 2
        self.winner = 0
        self.full = False
        self.history = []

    def is_valid(self, action):
        return action >= 0 and action < 7 and self.heights[action] < 6
    def charAt(self, col, row):
        return ' ' if row < 0 or row > 5 or col < 0 or col > 6 or self.board[0, row, col] == 1 else 'X' if self.board[1, row, col] == 1 else 'O'
    
    def setAt(self, col):
        row = self.heights[col]
        self.board[0, row, col] = 0
        self.board[self.player, row, col] = 1
        self.heights[col] += 1

    def move(self, action):
        if not self.is_valid(action):
            raise ValueError(f'Column {action} is already filled.')
        if self.winner != 0:
            raise ValueError(f'Game is already finished.')

        self.setAt(action)

        row = self.heights[action] - 1
        pattern = ''.join(['X' if self.player == 1 else 'O'] * 4)

        # check horizontal win
        if pattern in ''.join([self.charAt(col, row) for col in range(7)]):
            self.winner = self.player
        # check vertical win
        elif pattern in ''.join(self.charAt(action, r) for r in range(self.heights[action])):
            self.winner = self.player
        else:
            # check diagonal win ll to ur
            indices = [(action + (x - 4), row + (x - 4)) for x in range(7)]
            diagonal = ''.join([self.charAt(col, row) for col, row in indices])
            if pattern in diagonal:
                self.winner = self.player
            else:
                # check diagonal win ul to lr
                indices = [(action + (x - 3), row - (x - 3)) for x in range(7)]
                diagonal = ''.join([self.charAt(col, row) for col, row in indices])
                if pattern in diagonal:
                    self.winner = self.player

        self.player, self.opponent = self.opponent, self.player
        self.history.append(action)
        self.full = all(self.heights[i] == 6 for i in range(7))

    @property
    def state(self):
        clone = self.board.clone()
        if self.player == 1: 
            return clone
        clone[1], clone[2] = clone[2].clone(), clone[1].clone()
        return clone
        

    def render(self):
        for row in reversed(range(6)):
            signs = [self.charAt(col, row) for col in range(7)]
            print(f"|{''.join(signs)}|")

@torch.no_grad()
def validate(model : nn.Module, games, alpha = 0):
    train = model.training
    model.eval()

    crosswins = 0
    crossdraws = 0
    crosslosses = 0
    circlewins = 0
    circledraws = 0
    circlelosses = 0
    qplayer = 2

    for _ in range(games):
        done = False
        qplayer = 3 - qplayer
        env = Connect4()
        while not done:
            if qplayer == env.player or random.uniform(0, 1) < alpha:
                q = model(env.state).squeeze()
                action = max([a for a in range(7) if env.is_valid(a)], key = lambda x: q[x])
            else:
                action = random.choice([a for a in range(7) if env.is_valid(a)])

            env.move(action)

            if env.winner != 0:
                if env.winner == qplayer:
                    if qplayer == 1:
                        crosswins += 1
                    else:
                        circlewins += 1
                else:
                    if qplayer == 1:
                        crosslosses += 1
                    else:
                        circlelosses += 1
                done = True
            elif env.full:
                if qplayer == 1:
                    crossdraws += 1
                else:
                    circledraws += 1
                done = True

    if train:
        model.train()

    return crosswins, crossdraws, crosslosses, circlewins, circledraws, circlelosses
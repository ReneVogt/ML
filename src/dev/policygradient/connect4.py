import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Connect4:   
    def __init__(self):
        self.board = [0]*42
        self.columnState = [0]*7
        self.player = 1
        self.opponent = 2
        self.winner = 0
        self.full = False

    def is_valid(self, action):
        return action >= 0 and action < 7 and self.columnState[action] < 6
    
    def toindex(column, row):
        return column * 6 + 5 - row

    def move(self, action):
        if not self.is_valid(action):
            raise ValueError(f'Column {action} is already filled.')
        if self.winner != 0:
            raise ValueError(f'Game is already finished.')

        self.board[Connect4.toindex(action, self.columnState[action])] = self.player
        self.columnState[action] += 1

        def iswon():
            l = r = u = d = ul = ur = ll = lr = True
            h = v = d1 = d2 = 0
            row = self.columnState[action]-1
            col = action
            for i in range(1, 4):                
                if l and col - i >= 0 and self.board[Connect4.toindex(col-i, row)] == self.player:
                    h += 1
                    if h == 3:
                        return True
                else:
                    l = False
                if r and col + i < 7 and self.board[Connect4.toindex(col+i, row)] == self.player:
                    h += 1
                    if h == 3:
                        return True
                else:
                    r = False
                if u and row + i < 6 and self.board[Connect4.toindex(col, row+i)] == self.player:
                    v += 1
                    if v == 3:
                        return True
                else:
                    u = False
                if d and row - i >= 0 and self.board[Connect4.toindex(col, row-i)] == self.player:
                    v += 1
                    if v == 3:
                        return True
                else:
                    d = False
                if ul and row + i < 6 and col - i >= 0 and self.board[Connect4.toindex(col-i, row+i)] == self.player:
                    d1 += 1
                    if d1 == 3:
                        return True
                else:
                    ul = False
                if lr and row - i >= 0 and col + i < 7 and self.board[Connect4.toindex(col+i, row-i)] == self.player:
                    d1 += 1
                    if d1 == 3:
                        return True
                else:
                    lr = False
                if ur and row + i < 6 and col + i < 7 and self.board[Connect4.toindex(col+i, row+i)] == self.player:
                    d2 += 1
                    if d2 == 3:
                        return True
                else:
                    ur = False
                if ll and row - i >= 0 and col - i >= 0 and self.board[Connect4.toindex(col-i, row-i)] == self.player:
                    d2 += 1
                    if d2 == 3:
                        return True
                else:
                    ll = False

            return False
        
        if iswon():
            self.winner = self.player

        self.player, self.opponent = self.opponent, self.player
        self.full = sum(self.columnState) == 42

    @property
    def state(self):
        return F.one_hot(torch.tensor(self.board), 3).view(-1).float()

    def render(self):
        for row in reversed(range(6)):
            signs = [self.board[Connect4.toindex(col,row)] for col in range(7)]
            signs = [' ' if x == 0 else 'X' if x == 1 else 'O' for x in signs]
            print(f"|{''.join(signs)}|")
    

def sample(probs, env : Connect4):
    mask = torch.tensor([1 if env.is_valid(a) else 0 for a in range(7)])
    p = probs * mask
    p /= p.sum()
    return torch.multinomial(p, num_samples=1)

@torch.no_grad()
def generateEpisode(model : nn.Module):
    episode = []
    done = False
    train = model.training
    model.eval()
    env = Connect4()
    
    while not done:
        state = env.state
        probs = model(state)

        action = sample(probs, env)
        
        try:
            env.move(action.item())
        except Exception as e:
            print(f'ERROR: {e}')
            env.render()
            print(probs)

        if env.winner != 0:
            reward = 1
            done = True
        elif env.full:
            reward = 0
            done = True
        else:
            reward = -0.1

        episode.append((state, action, reward))
    
    if train:
        model.train()
    return episode

@torch.no_grad()
def validate(model : nn.Module, games):
    train = model.training
    model.eval()

    playerNames = {1: 'Red', 2: 'Yellow'}

    for player in  [1, 2]:
        wins = draws = losses = 0
        for _ in range(games):
            env = Connect4()
            while env.winner == 0 and not env.full:
                if player == env.player:
                    probs = model(env.state)
                    action = max([a for a in range(7) if env.is_valid(a)], key = lambda x: probs[x])
                else:
                    action = random.choice([a for a in range(7) if env.is_valid(a)])
                
                try:
                    env.move(action)
                except Exception as e:
                    print(f'ERROR: {e}')
                    env.render()
                    print(probs)

            if env.winner == player:
                wins += 1
            elif env.winner == 0:
                draws += 1
            else:
                losses += 1
        print(f'{playerNames[player]}: {100*wins/games:.2f} {100*draws/games:.2f} {100*losses/games:.2f}')

    if train:
        model.train()
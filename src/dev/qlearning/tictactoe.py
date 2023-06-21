class TicTacToe:
    def __init__(self):
        self.board = [' ']*9
        self.player = 'X'
        self.opponent = 'O'

    def is_valid(self, action):
        return self.board[action] == ' '

    def is_won(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]
        return any(self.board[i] == self.board[j] == self.board[k] == self.opponent for i, j, k in lines)

    def is_full(self):
        return ' ' not in self.board
    
    def is_finished(self): 
        return self.is_full() or self.is_won()

    def move(self, action):
        if not self.is_valid(action):
            return
        self.board[action] = self.player
        self.player, self.opponent = self.opponent, self.player

    @property
    def state(self):
        return ''.join(self.board) #torch.tensor([[1 if c == ' ' else 0, 1 if c == self.player else 0, 1 if c == self.opponent else 0] for c in self.board], dtype = torch.float32).view(-1)

    def render(self):
        print(f"|{''.join(self.board[0:3])}|")
        print(f"|{''.join(self.board[3:6])}|")
        print(f"|{''.join(self.board[6:])}|")
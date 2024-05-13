import torch as T

class TicTacToeBoard():
    """
    Represents a Tic Tac Toe game board.
    """
    def __init__(self):
        self.board = [' ']*9
        self.player = 'X'
        self.opponent = 'O'

    @property
    def won(self) -> bool:
        """
        Indicates this game was won by the last move.
        """
        return self.is_won()
    
    @property
    def draw(self) -> bool:
        """
        Indicates if this game ended as a draw.
        """
        return self.is_full()

    @property
    def finished(self) -> bool:
        """
        Indicates if this game finished.
        """
        return self.won or self.draw

    @property 
    def validMovesMask(self) -> T.Tensor:
        """
        Returns a mask tensor indicating valid and invalid moves.
        """
        return T.tensor([self.is_valid(a) for a in range(9)], dtype=bool)

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
        """
        Performs the given action on the board.
        """
        if self.finished:
            raise RuntimeError('Game already finished.')
        if not self.is_valid(action):
            raise IndexError('Position already occupied')
        
        self.board[action] = self.player
        self.player, self.opponent = self.opponent, self.player

    def reset(self):
        """
        Resets the game board.
        """
        self.board = [' ']*9
        self.player = 'X'
        self.opponent = 'O'

    @property
    def stateTensor(self):
        return T.tensor([[1 if c == ' ' else 0, 1 if c == self.player else 0, 1 if c == self.opponent else 0] for c in self.board], dtype = T.float32).view(-1)
    @property
    def state(self):
        return ''.join(self.board)

    def render(self):
        """
        Renders the board as console output.
        """
        print(f"|{''.join(self.board[0:3])}|")
        print(f"|{''.join(self.board[3:6])}|")
        print(f"|{''.join(self.board[6:])}|")
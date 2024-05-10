class Connect4BoardError(Exception):
    """
    Base class for errors raised by the
    Connect4Board class
    """
    def __init__(self, message) -> None:
        super().__init__(message)

class GameFinishedError(Connect4BoardError):
    """
    Error raised by the Connect4Board class
    when move() is called after a game already
    finished.
    """
    def __init__(self) -> None:
        super().__init__("The game was already finished.")
class ColumnFullError(Connect4BoardError):
    """
    Error raised by the Connect4Board class
    when move() is called for a column that is
    already full.
    """
    def __init__(self, column : int) -> None:
        super().__init__(f'Column {column} is already full.')

    
class Connect4Board:
    PLAYER1 = 1
    PLAYER2 = 2
    EMPTY = 0

    @property
    def Player(self) -> int:
        return self._player
    @property
    def ValidMoves(self) -> list[int]:
        return self._validmoves
    @property
    def Winner(self) -> int:
        return self._winner
    @property
    def Full(self) -> bool:
        return len(self._validmoves) == 0
    @property
    def Finished(self) -> bool:
        return self.Full or self.Winner != Connect4Board.EMPTY
    @property
    def gameKey(self) -> str:
        return ''.join(str(action) for action in self._history)

    def __init__(self) -> None:
        self._player = Connect4Board.PLAYER1        
        self._winner = Connect4Board.EMPTY
        self._history = []
        self._board = [[Connect4Board.EMPTY for _ in range(6)] for _ in range(7)]
        self._heights = [0, 0, 0, 0, 0, 0, 0]
        self._validmoves = [0, 1, 2, 3, 4, 5, 6]

    def __getitem__(self, position : tuple[int, int]) -> int:
        """
        Returns the player at the given position.

        Parameters:
        position (tuple[int,int]): The board column (zero-indexed, 0 left, 6 right) and row (zero-indexed, 0 bottom, 5 top) to look at.

        Returns:
        int: The player index at the given position (EMPTY, PLAYER1 or PLAYER2)
        """
        if not isinstance(position, tuple) or len(position) != 2:
            raise ValueError("Index must be a tuple of two integers.")
        column, row = position
        return self._board[column][row]

    def move(self, action) -> None:
        if self.Finished:
            raise GameFinishedError()
        if action < 0 or action > 6:
            raise IndexError("The action index must be greater than or equal to zero and less than 7.")
        
        row = self._heights[action]
        if row >= 6:
            raise ColumnFullError(action)
        
        self._board[action][row] = self._player
        self._heights[action] = row + 1
        if row == 5:
            self._validmoves.remove(action)

        self._winner = self._getWinner(action, row, self._player)
        self._history.append(action)
        self._player = Connect4Board.PLAYER2 if self._player == Connect4Board.PLAYER1 else Connect4Board.PLAYER1

    def _playerAt(self, column, row) -> int:
        if column < 0 or column > 6:
            return Connect4Board.EMPTY
        if row < 0 or row >= self._heights[column]:
            return Connect4Board.EMPTY
        return self._board[column][row]
    
    def _getWinner(self, column, row, player) -> int:
        # horizontal line
        line = 0
        if self._playerAt(column-1, row) == player:
            line += 1
            if self._playerAt(column-2, row) == player:
                line += 1
                if self._playerAt(column-3, row) == player:
                    return player
        if self._playerAt(column+1, row) == player:
            line += 1
            if line == 3:
                return player
            if self._playerAt(column+2, row) == player:
                line += 1
                if line == 3:
                    return player
                if self._playerAt(column+3, row) == player:
                    return player
                
        # vertical line
        line = 0
        if self._playerAt(column, row-1) == player:
            line += 1
            if self._playerAt(column, row-2) == player:
                line += 1
                if self._playerAt(column, row-3) == player:
                    return player
        if self._playerAt(column, row+1) == player:
            line += 1
            if line == 3:
                return player
            if self._playerAt(column, row+2) == player:
                line += 1
                if line == 3:
                    return player
                if self._playerAt(column, row+3) == player:
                    return player
                
        # diagonal line (ul to lr)
        line = 0
        if self._playerAt(column-1, row+1) == player:
            line += 1
            if self._playerAt(column-2, row+2) == player:
                line += 1
                if self._playerAt(column-3, row+3) == player:
                    return player
        if self._playerAt(column+1, row-1) == player:
            line += 1
            if line == 3:
                return player
            if self._playerAt(column+2, row-2) == player:
                line += 1
                if line == 3:
                    return player
                if self._playerAt(column+3, row-3) == player:
                    return player

        # diagonal line (ur to ll)
        line = 0
        if self._playerAt(column+1, row+1) == player:
            line += 1
            if self._playerAt(column+2, row+2) == player:
                line += 1
                if self._playerAt(column+3, row+3) == player:
                    return player
        if self._playerAt(column-1, row-1) == player:
            line += 1
            if line == 3:
                return player
            if self._playerAt(column-2, row-2) == player:
                line += 1
                if line == 3:
                    return player
                if self._playerAt(column-3, row-3) == player:
                    return player

        return Connect4Board.EMPTY
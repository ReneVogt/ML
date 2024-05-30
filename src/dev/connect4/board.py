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
    def Winner(self) -> int:
        return self._winner
    @property
    def Full(self) -> bool:
        return self._full
    @property
    def Finished(self) -> bool:
        return self.Winner != Connect4Board.EMPTY or self.Full
    @property
    def stateKey(self) -> int:
        return self._state
    @property
    def gameKey(self) -> str:
        return ''.join(str(action) for action in self._history)

    def __init__(self) -> None:
        self._player = Connect4Board.PLAYER1        
        self._winner = Connect4Board.EMPTY
        self._full = False
        self._history = []
        self._state = 0

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
        if column < 0:
            raise IndexError("Column index must be greater than or equal to 0.")
        if column > 6:
            raise IndexError("Column index must be less than 7.")
        if row < 0:
            raise IndexError("Row index must be greater than or equal to 0.")
        if row > 5:
            raise IndexError("Row index must be less than 6.")
        return self._getPlayerAt(column, row)

    def move(self, action : int) -> None:
        if self.Finished:
            raise GameFinishedError()
        if action < 0 or action > 6:
            raise IndexError("The action index must be greater than or equal to zero and less than 7.")
        
        row = self._getColumnHeight(action)
        if row >= 6:
            raise ColumnFullError(action)
        
        self._setPlayerAt(action, row, self._player)
        self._setColumnHeight(action, row + 1)

        if row == 5:
            self._full = not any(self.is_valid(a) for a in range(7))

        self._winner = self._getWinner(action, row, self._player)
        self._history.append(action)
        self._player = Connect4Board.PLAYER2 if self._player == Connect4Board.PLAYER1 else Connect4Board.PLAYER1

    def is_valid(self, action : int) -> bool:
        return action >= 0 and action < 7 and self._getColumnHeight(action) < 6

    def undo(self) -> None:
        if len(self._history) == 0:
            return
        action = self._history.pop()
        row = self._getColumnHeight(action) - 1
        self._setPlayerAt(action, row, Connect4Board.PLAYER1)
        self._setColumnHeight(action, row)
        self._winner = Connect4Board.EMPTY
        self._full = False
    
    def clone(self) -> 'Connect4Board':
        board = Connect4Board()
        board._state = self._state
        board._winner = self._winner
        board._history = self._history.copy()
        board._player = self._player
        board._full = self._full
        return board
    
    def _getColumnHeight(self, column : int) -> int:
        return (self._state >> (9 * column)) & 7
    def _setColumnHeight(self, column : int, height : int) -> None:
        state = self._state & ~(7 << (9 * column))
        self._state = state | (height << (9 * column))               
    def _getPlayerAt(self, column : int, row : int) -> int:
        if column < 0 or column > 6:
            return Connect4Board.EMPTY
        if row < 0 or row >= self._getColumnHeight(column):
            return Connect4Board.EMPTY
        return ((self._state >> (9 * column + row + 3)) & 1) + 1
    def _setPlayerAt(self, column : int, row : int, player : int) -> int:
        state = self._state & ~(1 << 9 * column + row + 3)
        if player == Connect4Board.PLAYER2:
            state |= 1 << (9 * column + row + 3)
        self._state = state
    
    def _getWinner(self, column, row, player) -> int:
        # horizontal line
        line = 0
        if self._getPlayerAt(column-1, row) == player:
            line += 1
            if self._getPlayerAt(column-2, row) == player:
                line += 1
                if self._getPlayerAt(column-3, row) == player:
                    return player
        if self._getPlayerAt(column+1, row) == player:
            line += 1
            if line == 3:
                return player
            if self._getPlayerAt(column+2, row) == player:
                line += 1
                if line == 3:
                    return player
                if self._getPlayerAt(column+3, row) == player:
                    return player
                
        # vertical line
        line = 0
        if self._getPlayerAt(column, row-1) == player:
            line += 1
            if self._getPlayerAt(column, row-2) == player:
                line += 1
                if self._getPlayerAt(column, row-3) == player:
                    return player
        if self._getPlayerAt(column, row+1) == player:
            line += 1
            if line == 3:
                return player
            if self._getPlayerAt(column, row+2) == player:
                line += 1
                if line == 3:
                    return player
                if self._getPlayerAt(column, row+3) == player:
                    return player
                
        # diagonal line (ul to lr)
        line = 0
        if self._getPlayerAt(column-1, row+1) == player:
            line += 1
            if self._getPlayerAt(column-2, row+2) == player:
                line += 1
                if self._getPlayerAt(column-3, row+3) == player:
                    return player
        if self._getPlayerAt(column+1, row-1) == player:
            line += 1
            if line == 3:
                return player
            if self._getPlayerAt(column+2, row-2) == player:
                line += 1
                if line == 3:
                    return player
                if self._getPlayerAt(column+3, row-3) == player:
                    return player

        # diagonal line (ur to ll)
        line = 0
        if self._getPlayerAt(column+1, row+1) == player:
            line += 1
            if self._getPlayerAt(column+2, row+2) == player:
                line += 1
                if self._getPlayerAt(column+3, row+3) == player:
                    return player
        if self._getPlayerAt(column-1, row-1) == player:
            line += 1
            if line == 3:
                return player
            if self._getPlayerAt(column-2, row-2) == player:
                line += 1
                if line == 3:
                    return player
                if self._getPlayerAt(column-3, row-3) == player:
                    return player

        return Connect4Board.EMPTY
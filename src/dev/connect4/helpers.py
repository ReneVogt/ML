from connect4 import Connect4Board
import datetime

def log(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

def render(board : Connect4Board):
    print('╔═══╤═══╤═══╤═══╤═══╤═══╤═══╗')
    for row in range(5, -1, -1):
        line = ''
        for col in range(7):
            if col == 0:
                line = '║ '
            player = board[col, row]
            line += 'X' if player == Connect4Board.PLAYER1 else 'O' if player == Connect4Board.PLAYER2 else ' '
            line += ' │ ' if col < 6 else ' ║'
        print(line)
        if row > 0:
            print('╟───┼───┼───┼───┼───┼───┼───╢')
    print('╚═══╧═══╧═══╧═══╧═══╧═══╧═══╝')

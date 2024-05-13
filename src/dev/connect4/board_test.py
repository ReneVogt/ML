import sys
sys.path.append('../../')

import unittest
from connect4.board import Connect4Board, GameFinishedError, ColumnFullError

class TestConnect4Board_Basics_FullColumns_Draw(unittest.TestCase):
    def test_CorrectIndexer(self):
        env = Connect4Board()
        env.move(0)
        env.move(6)
        env.move(6)
        self.assertEqual(Connect4Board.PLAYER1, env[0, 0])

        self.assertEqual(Connect4Board.EMPTY, env[0, 1])
        self.assertEqual(Connect4Board.EMPTY, env[1, 0])
        self.assertEqual(Connect4Board.EMPTY, env[1, 1])
        
        self.assertEqual(Connect4Board.PLAYER2, env[6, 0])
        self.assertEqual(Connect4Board.PLAYER1, env[6, 1])
        self.assertEqual(Connect4Board.EMPTY, env[6, 2])
        

    def test_FullBoard_CorrectKeyAndStates(self):
        moves = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
                 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
                 4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
                 6, 6, 6, 6, 6, 6]
        
        player = Connect4Board.PLAYER1
        opponent = Connect4Board.PLAYER2

        env = Connect4Board()

        for i in range(len(moves)):
            action = moves[i]
            self.assertEqual(env.Player, player, "Wrong current player.")
            env.move(action)
            player, opponent = opponent, player

            if i < 10:
                self.assertEqual([0, 1, 2, 3, 4, 5, 6], env.ValidMoves)
            elif i < 11:
                self.assertEqual([0, 2, 3, 4, 5, 6], env.ValidMoves)
            elif i < 22:
                self.assertEqual([2, 3, 4, 5, 6], env.ValidMoves)
            elif i < 23:
                self.assertEqual([2, 4, 5, 6], env.ValidMoves)
            elif i < 34:
                self.assertEqual([4, 5, 6], env.ValidMoves)
            elif i < 35:
                self.assertEqual([4, 6], env.ValidMoves)
            elif i < 41:
                self.assertEqual([6], env.ValidMoves)

            if i < len(moves)-1:
                self.assertEqual(Connect4Board.EMPTY, env.Winner)
                self.assertFalse(env.Full)
                self.assertFalse(env.Finished)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertTrue(env.Full)
        self.assertTrue(env.Finished)
        self.assertEqual([], env.ValidMoves)
        self.assertEqual("010101101010232323323232454545545454666666", env.gameKey)

        with self.assertRaises(GameFinishedError):
            env.move(0)

    def test_FullColumn0_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [0, 0, 0, 0, 0, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([1, 2, 3, 4, 5, 6], env.ValidMoves)
        self.assertEqual("000000", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(0)

    def test_FullColumn1_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [1, 1, 1, 1, 1, 1]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 2, 3, 4, 5, 6], env.ValidMoves)
        self.assertEqual("111111", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(1)

    def test_FullColumn2_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [2, 2, 2, 2, 2, 2]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 1, 3, 4, 5, 6], env.ValidMoves)
        self.assertEqual("222222", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(2)

    def test_FullColumn3_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [3, 3, 3, 3, 3, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 1, 2, 4, 5, 6], env.ValidMoves)
        self.assertEqual("333333", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(3)

    def test_FullColumn4_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [4, 4, 4, 4, 4, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 1, 2, 3, 5, 6], env.ValidMoves)
        self.assertEqual("444444", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(4)

    def test_FullColumn5_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [5, 5, 5, 5, 5, 5]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 1, 2, 3, 4, 6], env.ValidMoves)
        self.assertEqual("555555", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(5)

    def test_FullColumn6_CorrectValidMovesKeyAndException(self):
        env = Connect4Board()
        for action in [6, 6, 6, 6, 6, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.EMPTY, env.Winner)
        self.assertFalse(env.Full)
        self.assertFalse(env.Finished)
        self.assertEqual([0, 1, 2, 3, 4, 5], env.ValidMoves)
        self.assertEqual("666666", env.gameKey)

        with self.assertRaises(ColumnFullError):
            env.move(6)

class TestConnect4Board_Player1WinsHorizontally(unittest.TestCase):
    def test_LowerLeft_LastLeft(self):
        env = Connect4Board()
        for action in [1, 6, 2, 6, 3, 6, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastRight(self):
        env = Connect4Board()
        for action in [0, 6, 1, 6, 2, 6, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastLeft(self):
        env = Connect4Board()
        for action in [6, 0, 5, 0, 4, 0, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastRight(self):
        env = Connect4Board()
        for action in [3, 0, 4, 0, 5, 0, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperLeft_LastLeft(self):
        env = Connect4Board()
        for action in [0, 1, 0, 1, 1, 0, 1, 0, 2, 3, 2, 3, 3, 2, 3, 2, 0, 1, 2, 3,
                       3, 6, 2, 6, 1, 6, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperLeft_LastRight(self):
        env = Connect4Board()
        for action in [0, 1, 0, 1, 1, 0, 1, 0, 2, 3, 2, 3, 3, 2, 3, 2, 0, 1, 2, 3,
                       0, 6, 1, 6, 2, 6, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight_LastLeft(self):
        env = Connect4Board()
        for action in [3, 4, 3, 4, 4, 3, 4, 3, 5, 6, 5, 6, 6, 5, 6, 5, 3, 4, 5, 6,
                       6, 0, 5, 0, 4, 0, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

    def test_UpperRight_LastRight(self):
        env = Connect4Board()
        for action in [3, 4, 3, 4, 4, 3, 4, 3, 5, 6, 5, 6, 6, 5, 6, 5, 3, 4, 5, 6,
                       3, 0, 4, 0, 5, 0, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [1, 2, 1, 2, 3, 4, 3, 4, 
                       4, 6, 3, 6, 2, 6, 1]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [1, 2, 1, 2, 3, 4, 3, 4, 
                       1, 6, 2, 6, 3, 6, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

class TestConnect4Board_Player1WinsVertically(unittest.TestCase):
    def test_LowerLeft(self):
        env = Connect4Board()
        for action in [0, 6, 0, 6, 0, 6, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight(self):
        env = Connect4Board()
        for action in [6, 0, 6, 0, 6, 0, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

    def test_UpperLeft(self):
        env = Connect4Board()
        for action in [1, 0, 1, 0,
                       0, 2, 0, 3, 0, 4, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight(self):
        env = Connect4Board()
        for action in [5, 6, 5, 6,
                       6, 0, 6, 1, 6, 2, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                

    def test_MiddleCenter(self):
        env = Connect4Board()
        for action in [3, 4, 4, 3, 4, 3, 4, 3, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

class TestConnect4Board_Player1WinsDiegonalUpperLeftToLowerRight(unittest.TestCase):    
    def test_UpperLeft_LastLeft(self):
        env = Connect4Board()
        for action in [0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 3, 2, 2, 1, 1, 0, 0, 3, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperLeft_LastRight(self):
        env = Connect4Board()
        for action in [0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 4, 2, 2, 1, 1, 0, 0, 6, 0, 6, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastLeft(self):
        env = Connect4Board()
        for action in [6, 5, 5, 4, 3, 4, 4, 3, 2, 3, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastRight(self):
        env = Connect4Board()
        for action in [3, 5, 5, 4, 3, 4, 4, 3, 3, 0, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [1, 2, 3, 4, 4, 3, 3, 2, 1, 2, 2, 1, 1, 2, 1]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [1, 2, 3, 4, 2, 3, 3, 2, 1, 1, 2, 5, 1, 5, 1, 5, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

class TestConnect4Board_Player1WinsDiegonalUpperRightToLowerLeft(unittest.TestCase):    
    def test_UpperRight_LastRight(self):
        env = Connect4Board()
        for action in [6, 5, 4, 3, 6, 5, 4, 3, 5, 6, 3, 4, 4, 5, 5, 6, 6, 3, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight_LastLeft(self):
        env = Connect4Board()
        for action in [6, 5, 4, 3, 6, 5, 4, 3, 5, 6, 2, 4, 4, 5, 5, 6, 6, 0, 6, 0, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastRight(self):
        env = Connect4Board()
        for action in [0, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastLeft(self):
        env = Connect4Board()
        for action in [3, 1, 1, 2, 3, 2, 2, 3, 3, 6, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [5, 4, 3, 2, 2, 3, 3, 4, 5, 4, 4, 5, 5, 4, 5]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [5, 4, 3, 2, 4, 3, 3, 4, 5, 5, 4, 1, 5, 1, 5, 1, 2]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER1, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

class TestConnect4Board_Player2WinsHorizontally(unittest.TestCase):
    def test_LowerLeft_LastLeft(self):
        env = Connect4Board()
        for action in [6, 3, 6, 2, 6, 1, 5, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastRight(self):
        env = Connect4Board()
        for action in [6, 0, 6, 1, 6, 2, 5, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastLeft(self):
        env = Connect4Board()
        for action in [0, 6, 0, 5, 0, 4, 1, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastRight(self):
        env = Connect4Board()
        for action in [0, 3, 0, 4, 0, 5, 1, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

    def test_UpperLeft_LastLeft(self):
        env = Connect4Board()
        for action in [0, 1, 0, 1, 1, 0, 1, 0, 2, 3, 2, 3, 3, 2, 3, 2, 0, 1, 2, 3,
                       6, 3, 6, 2, 6, 1, 5, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)

    def test_UpperLeft_LastRight(self):
        env = Connect4Board()
        for action in [0, 1, 0, 1, 1, 0, 1, 0, 2, 3, 2, 3, 3, 2, 3, 2, 0, 1, 2, 3,
                       6, 0, 6, 1, 6, 2, 5, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight_LastLeft(self):
        env = Connect4Board()
        for action in [3, 4, 3, 4, 4, 3, 4, 3, 5, 6, 5, 6, 6, 5, 6, 5, 3, 4, 5, 6,
                       0, 6, 0, 5, 0, 4, 1, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                

    def test_UpperRight_LastRight(self):
        env = Connect4Board()
        for action in [3, 4, 3, 4, 4, 3, 4, 3, 5, 6, 5, 6, 6, 5, 6, 5, 3, 4, 5, 6,
                       0, 3, 0, 4, 0, 5, 1, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [1, 2, 1, 2, 3, 4, 3, 4, 
                       6, 4, 6, 3, 6, 2, 5, 1]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [1, 2, 1, 2, 3, 4, 3, 4, 
                       6, 1, 6, 2, 6, 3, 5, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

class TestConnect4Board_Player2WinsVertically(unittest.TestCase):
    def test_LowerLeft(self):
        env = Connect4Board()
        for action in [6, 0, 6, 0, 6, 0, 5, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight(self):
        env = Connect4Board()
        for action in [0, 6, 0, 6, 0, 6, 1, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

    def test_UpperLeft(self):
        env = Connect4Board()
        for action in [0, 1, 0, 1,
                       2, 0, 3, 0, 4, 0, 4, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight(self):
        env = Connect4Board()
        for action in [6, 5, 6, 5, 
                       0, 6, 0, 6, 1, 6, 2, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                

    def test_MiddleCenter(self):
        env = Connect4Board()
        for action in [3, 4, 4, 3, 5, 3, 6, 3, 6, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

class TestConnect4Board_Player2WinsDiegonalUpperLeftToLowerRight(unittest.TestCase):    
    def test_UpperLeft_LastLeft(self):
        env = Connect4Board()
        for action in [6, 0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 3, 2, 2, 1, 1, 0, 0, 3, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperLeft_LastRight(self):
        env = Connect4Board()
        for action in [5, 0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 4, 2, 2, 1, 1, 0, 0, 6, 0, 6, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastLeft(self):
        env = Connect4Board()
        for action in [0, 6, 5, 5, 4, 3, 4, 4, 3, 2, 3, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerRight_LastRight(self):
        env = Connect4Board()
        for action in [1, 3, 5, 5, 4, 3, 4, 4, 3, 3, 0, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [6, 1, 2, 3, 4, 4, 3, 3, 2, 1, 2, 2, 1, 1, 2, 1]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [6, 1, 2, 3, 4, 2, 3, 3, 2, 1, 1, 2, 5, 1, 5, 1, 5, 4]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

class TestConnect4Board_Player2WinsDiegonalUpperRightToLowerLeft(unittest.TestCase):    
    def test_UpperRight_LastRight(self):
        env = Connect4Board()
        for action in [0, 6, 5, 4, 3, 6, 5, 4, 3, 5, 6, 3, 4, 4, 5, 5, 6, 6, 3, 6]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_UpperRight_LastLeft(self):
        env = Connect4Board()
        for action in [1, 6, 5, 4, 3, 6, 5, 4, 3, 5, 6, 2, 4, 4, 5, 5, 6, 6, 0, 6, 0, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastRight(self):
        env = Connect4Board()
        for action in [6, 0, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)

    def test_LowerLeft_LastLeft(self):
        env = Connect4Board()
        for action in [5, 3, 1, 1, 2, 3, 2, 2, 3, 3, 6, 0]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastRight(self):
        env = Connect4Board()
        for action in [0, 5, 4, 3, 2, 2, 3, 3, 4, 5, 4, 4, 5, 5, 4, 5]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)                        

    def test_MiddleCenter_LastLeft(self):
        env = Connect4Board()
        for action in [0, 5, 4, 3, 2, 4, 3, 3, 4, 5, 5, 4, 1, 5, 1, 5, 1, 2]:
            env.move(action)

        self.assertEqual(Connect4Board.PLAYER2, env.Winner)
        self.assertFalse(env.Full)
        self.assertTrue(env.Finished)        

if __name__ == '__main__':
    unittest.main()
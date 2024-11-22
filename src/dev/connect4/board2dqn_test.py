import unittest

import torch as T
import torch.nn as nn

from connect4.board import Connect4Board
from connect4.board2dqn import createStateTensor, getBestAction

class TestModel(nn.Module):
    def __init__(self, forwardCallback):
        super(TestModel, self).__init__()
        self.forwardCallback = forwardCallback

    def forward(self, x):
        return self.forwardCallback(x)

class TestBoard2Dqn_StateTennor(unittest.TestCase):
    def test_EmptyBoard_ZeroTensor(self):
        expectedTensor = T.zeros(3, 6, 7)
        expectedTensor[0,:,:] = 1
        board = Connect4Board()
        result = createStateTensor(board)
        self.assertTrue(T.equal(result, expectedTensor))

    def test_UsedBoard_Player1(self):
        board = Connect4Board()
        expectedTensor = T.zeros(3, 6, 7)
        expectedTensor[0,:,:] = 1

        board.move(0)
        expectedTensor[0, 0, 0] = 0
        expectedTensor[1, 0, 0] = 1

        board.move(1)
        expectedTensor[0, 0, 1] = 0
        expectedTensor[2, 0, 1] = 1

        board.move(1)
        expectedTensor[0, 1, 1] = 0
        expectedTensor[1, 1, 1] = 1 

        board.move(2)
        expectedTensor[0, 0, 2] = 0
        expectedTensor[2, 0, 2] = 1
        
        result = createStateTensor(board)
        self.assertTrue(T.equal(result, expectedTensor))

    def test_UsedBoard_Player2(self):
        board = Connect4Board()
        expectedTensor = T.zeros(3, 6, 7)
        expectedTensor[0,:,:] = 1

        board.move(0)
        expectedTensor[0, 0, 0] = 0
        expectedTensor[2, 0, 0] = 1 

        board.move(1)
        expectedTensor[0, 0, 1] = 0
        expectedTensor[1, 0, 1] = 1 

        board.move(1)
        expectedTensor[0, 1, 1] = 0
        expectedTensor[2, 1, 1] = 1 


        board.move(2)
        expectedTensor[0, 0, 2] = 0
        expectedTensor[1, 0, 2] = 1 


        board.move(2)
        expectedTensor[0, 1, 2] = 0
        expectedTensor[2, 1, 2] = 1

        result = createStateTensor(board)
        self.assertTrue(T.equal(result, expectedTensor))

class TestBoard2Dqn_BestAction(unittest.TestCase):
    def test_BestAction_Player1(self):
        env = Connect4Board()
        for move in [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1]:
            env.move(move)
        
        model = TestModel(lambda x: T.Tensor([[10, 7, 8, 6, 5, 5, 5.5]]))

        action = getBestAction(model, env)
        self.assertEqual(action, 2)


if __name__ == '__main__':
    unittest.main()
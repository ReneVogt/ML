import unittest
import training
import torch
import torch.nn as nn
import random
from connect4 import Connect4Board

class TestModel(nn.Module):
    def __init__(self, forwardCallback):
        super(TestModel, self).__init__()
        self.forwardCallback = forwardCallback

    def forward(self, x):
        return self.forwardCallback(x)

class TestTraining_MoveSampling(unittest.TestCase):
    def test_SingleValidMove_EpsilonZero(self):
        validMoves = [3]
        qvalues = torch.ones(7, dtype=torch.float32)
        result = training.sampleMove(qvalues, validMoves, 0)
        self.assertEqual(result, 3)

    def test_SingleValidMove_EpsilonOne(self):
        validMoves = [3]
        qvalues = torch.ones(7, dtype=torch.float32)
        result = training.sampleMove(qvalues, validMoves, 1)
        self.assertEqual(result, 3)

    def test_TakingRandomMove(self):
        validMoves = [4, 5, 6]
        qvalues = torch.zeros(7, dtype=torch.float32)
        qvalues[5] = 1
        
        # seeding random to 0
        # so random.uniform() returns 0.8444218515250481
        # and random.choice afterwards returns the middle element
        random.seed(0)
        torch.manual_seed(0)
        result = training.sampleMove(qvalues, validMoves, 0.84443)
        self.assertEqual(result, 5)

    def test_TakingSampledMove(self):
        validMoves = [3, 4, 5, 6]
        qvalues = torch.zeros(7, dtype=torch.float32)
        qvalues[4] = 1
        # seeding random to 0
        # so random.uniform() returns 0.8444218515250481
        # and multinomial afterwards returns the 5
        random.seed(0)
        torch.manual_seed(0)
        result = training.sampleMove(qvalues, validMoves, 0.843)
        self.assertEqual(result, 5)

class TestTraining_MoveSamplingInternal(unittest.TestCase):
    def test_SingleValidMove_EpsilonZero(self):
        board = Connect4Board()
        for action in [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
                 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
                 4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
                 6, 6, 6, 6, 6]:
            board.move(action)
        self.assertEqual(board.ValidMoves, [6])

        result = training._sampleMove(None, board, 0)
        self.assertEqual(result, 6)

    def test_SingleValidMove_EpsilonOne(self):
        board = Connect4Board()
        for action in [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
                 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
                 4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
                 6, 6, 6, 6, 6]:
            board.move(action)

        self.assertEqual(board.ValidMoves, [6])

        result = training._sampleMove(None, board, 1)
        self.assertEqual(result, 6)

    def test_TakingRandomMove(self):
        board = Connect4Board()
        for action in [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
                 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2]:
            board.move(action)
        
        self.assertEqual(board.ValidMoves, [4, 5, 6])

        # seeding random to 0
        # so random.uniform() returns 0.8444218515250481
        # and random.choice afterwards returns the middle element
        random.seed(0)
        torch.manual_seed(0)
        result = training._sampleMove(None, board, 0.84443)
        self.assertEqual(result, 5)

    def test_TakingSampledMove(self):
        board = Connect4Board()
        for action in [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
                 2, 2, 2, 2, 2, 2]:
            board.move(action)
        self.assertEqual(board.ValidMoves, [3, 4, 5, 6])

        qvalues = torch.zeros(7, dtype=torch.float32)
        qvalues[4] = 1        
        model = TestModel(lambda _: torch.stack([qvalues]))
        
        # seeding random to 0
        # so random.uniform() returns 0.8444218515250481
        # and multinomial afterwards returns the 5
        random.seed(0)
        torch.manual_seed(0)
        result = training._sampleMove(model, board, 0.843)
        self.assertEqual(result, 5)

class TestTraining_StateTennor(unittest.TestCase):
    def test_EmptyBoard_ZeroTensorOnes(self):
        expectedTensor = torch.zeros(3, 6, 7, dtype=torch.float32)
        expectedTensor[0, :, :] = 1
        board = Connect4Board()
        result = training.createStateTensor(board)
        self.assertTrue(torch.equal(result, expectedTensor))

    def test_UsedBoard_Player1(self):
        board = Connect4Board()
        expectedTensor = torch.zeros(3, 6, 7, dtype=torch.float32)
        expectedTensor[0, :, :] = 1

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
        
        result = training.createStateTensor(board)
        self.assertTrue(torch.equal(result, expectedTensor))

    def test_UsedBoard_Player2(self):
        board = Connect4Board()
        expectedTensor = torch.zeros(3, 6, 7, dtype=torch.float32)
        expectedTensor[0, :, :] = 1

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

        result = training.createStateTensor(board)
        self.assertTrue(torch.equal(result, expectedTensor))

if __name__ == '__main__':
    unittest.main()
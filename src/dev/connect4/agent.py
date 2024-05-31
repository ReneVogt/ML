import numpy as np
import torch as T
import torch as T
import torch.nn.functional as F
from connect4.dqn import Connect4Dqn
from connect4.board import Connect4Board
from connect4.board2dqn import getBestAction


def calculateReward(board : Connect4Board) -> float:
    if board.Winner != Connect4Board.EMPTY:
        return 1
    
    if board.Full:
        return -0.1 if board.Player == Connect4Board.PLAYER2 else 0.2
    
    return -0.1 if board.Player == Connect4Board.PLAYER2 else 0

_NEGINF = float('-inf')

class Connect4Agent():
    def __init__(self, lr : float = 0.001, 
                 epsilon : float = 0.5, epsilon_end : float = 0.01, epsilon_decay : float = 0,
                 batch_size : int = 512, batch_count : int = 1, memory_size : int = 0x10000, gamma : float = 0.9, targetUpdateInterval : int = 1) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.memory_size = memory_size
        self.memory_counter = 0
        self.iteration_counter = 0
        self.targetUpdateInterval = targetUpdateInterval

        self.evaluationModel = Connect4Dqn(lr)
        if self.targetUpdateInterval > 0:
            self.targetModel = Connect4Dqn(lr)
            self.targetModel.load_state_dict(self.evaluationModel.state_dict())
        else:
            self.targetModel = None

        self.cpu = T.device("cpu")
        self.gpu = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.state_memory = np.zeros((memory_size, 3, 6, 7), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, 3, 6, 7), dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.validmoves_memory = np.zeros((memory_size, 7), dtype=bool)

        self.losses = []

    @property
    def numberOfParameters(self) -> int:
        return sum(p.numel() for p in self.evaluationModel.parameters() if p.requires_grad)

    @T.no_grad()
    def store_transition(self, state, action, next_state, next_validmoves, terminal, reward):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.validmoves_memory[index] = next_validmoves

        self.memory_counter += 1

    @T.no_grad()
    def getTrainingAction(self, state : T.Tensor, validMoves : list[int]) -> int:
        if len(validMoves) == 1:
            return validMoves[0]
        
        if np.random.random() > self.epsilon:
            return getBestAction(self.evaluationModel, state, validMoves)
        
        return np.random.choice(validMoves)
    
    def learn(self) -> None:
        if self.memory_counter < self.batch_size:
            return
        
        self.evaluationModel.to(self.gpu)
        targetModel = self.evaluationModel
        if self.targetModel is not None:
            self.targetModel.to(self.gpu)
            targetModel = self.targetModel


        max_mem = min(self.memory_counter, self.memory_size)

        for _ in range(self.batch_count):
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.gpu)
            next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.gpu)
            action_batch = self.action_memory[batch]
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.gpu)
            terminal_batch = T.tensor(self.terminal_memory[batch])
            validmoves_batch = T.tensor(self.validmoves_memory[batch])

            targetModel.eval()
            with T.no_grad():
                q_next = targetModel.forward(next_state_batch)

            q_next[~validmoves_batch] = _NEGINF
            q_next[terminal_batch] = 0.0
            
            self.evaluationModel.train()
            q_eval = self.evaluationModel.forward(state_batch)[batch_index, action_batch]
            q_target = reward_batch - self.gamma*T.max(q_next, dim=1)[0]
            
            self.evaluationModel.optimizer.zero_grad()
            loss = self.evaluationModel.loss(q_eval, q_target).to(self.gpu)
            self.losses.append(loss.item())
            loss.backward()
            self.evaluationModel.optimizer.step()

            self.iteration_counter += 1
            if self.targetUpdateInterval > 0 and  self.iteration_counter % self.targetUpdateInterval == 0:
                self.updateTargetModel()

        self.evaluationModel.to(self.cpu)
        if self.targetModel is not None:
            self.targetModel.to(self.cpu)

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

    @T.no_grad()
    def updateTargetModel(self) -> None:
        if self.targetModel is not None:
            self.targetModel.load_state_dict(self.evaluationModel.state_dict())

    @T.no_grad()
    def loadCheckpoint(self, fileName : str) -> None:
        cp = T.load(f'{fileName}.nn');
        self.evaluationModel.load_state_dict(cp['model_state_dict']);
        self.updateTargetModel()

        self.losses = []

        print(f"Loaded checkpoint {fileName}.")

    @T.no_grad()
    def saveCheckpoint(self, fileName : str) -> None:
        self.evaluationModel.eval();
        try:
            T.save({
                'model_state_dict': self.evaluationModel.state_dict()
            }, f'{fileName}.nn');

            print(f"Checkpoint '{fileName}' saved.")
        except Exception as e:
            print(f"Failed to save checkpoint '{fileName}': {e}")

    @T.no_grad()
    def printStats(self) -> None:
        count = len(self.losses)
        if count < 1:
            return
        
        print(f'Average loss (last {count}): {sum(self.losses)/count}, last: {self.losses[-1]}, epsilon: {self.epsilon}')
        self.losses = []
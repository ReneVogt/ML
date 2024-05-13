import numpy as np
import torch as T
import torch.nn.functional as F
from dqn import TicTacToeModel

_NEGINF = float('-inf')

class TicTacToeAgent():
    def __init__(self, gamma : float, epsilon : float, lr : float, batch_size : int, memory_size : int, epsilon_end : float, epsilon_decay : float):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0

        self.model = TicTacToeModel(lr)
        self.state_memory = np.zeros((memory_size, 27), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, 27), dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.validmoves_memory = np.zeros((memory_size, 9), dtype=bool)

    @property
    def training(self):
        return self.model.training
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()

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
    def getTrainingAction(self, state : T.Tensor, validMoves : T.Tensor):
        if np.random.random() > self.epsilon:
            train = self.model.training
            self.model.eval()
            actions = self.model.forward(state)
            if train:
                self.model.train()
            actions[~validMoves] = _NEGINF
            return T.argmax(actions).item()

        v = [a for a in range(9) if validMoves[a]]
        return  np.random.choice(v)

    @T.no_grad()
    def getBestAction(self, state : T.Tensor, validMoves : T.Tensor):
        train = self.model.training
        self.model.eval()
        actions = self.model.forward(state)
        if train:
            self.model.train()
        actions[~validMoves] = _NEGINF
        return T.argmax(actions).item()

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        max_mem = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch])
        next_state_batch = T.tensor(self.next_state_memory[batch])
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch])
        terminal_batch = T.tensor(self.terminal_memory[batch])
        validmoves_batch = T.tensor(self.validmoves_memory[batch])        

        eval = not self.model.training

        self.model.eval()
        q_next = self.model.forward(next_state_batch)
        q_next[~validmoves_batch] = _NEGINF
        q_next[terminal_batch] = 0.0
        
        self.model.train()
        q_eval = self.model.forward(state_batch)[batch_index, action_batch]
        q_target = reward_batch - self.gamma*T.max(q_next, dim=1)[0]
        
        self.model.optimizer.zero_grad()
        loss = self.model.loss(q_eval, q_target)
        loss.backward()
        self.model.optimizer.step()

        if eval:
            self.model.eval()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end
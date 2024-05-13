import numpy as np
import torch as T
import torch as T
import torch.nn.functional as F
from connect4.dqn import Connect4Dqn
from connect4.board import Connect4Board

def createStateTensor(board : Connect4Board) -> T.Tensor:
    transposedBoard = T.tensor(board._board, dtype=T.int64).transpose(0, 1)
    onehot = F.one_hot(transposedBoard, num_classes=3).permute(2, 0, 1)
    if board.Player == Connect4Board.PLAYER2:
        onehot = onehot[T.tensor([0,2,1])]
    return onehot.float()

def _validMovesFromMask(validMovesMask : T.Tensor) -> list[int]:
    return [a for a in range(7) if validMovesMask[a]]

_NEGINF = float('-inf')

class Connect4Agent():
    def __init__(self, gamma : float, epsilon : float, lr : float, batch_size : int, memory_size : int, epsilon_end : float, epsilon_decay : float) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0

        self.model = Connect4Dqn(lr)
        self.state_memory = np.zeros((memory_size, 3, 6, 7), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, 3, 6, 7), dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.validmoves_memory = np.zeros((memory_size, 7), dtype=bool)

    @property
    def numberOfParameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
    def getTrainingAction(self, state : T.Tensor, validMovesMask : T.Tensor) -> int:
        if np.random.random() > self.epsilon:
            self.model.eval()
            actions = self.model.forward(state.unsqueeze(0))[0]
            actions[~validMovesMask] = _NEGINF
            return T.argmax(actions).item()

        validMoves = _validMovesFromMask(validMovesMask)
        return np.random.choice(validMoves)

    @T.no_grad()
    def getBestAction(self, state : T.Tensor, validMovesMask : T.Tensor) -> int:
        self.model.eval()
        actions = self.model.forward(state.unsqueeze(0))[0]
        actions[~validMovesMask] = _NEGINF
        return T.argmax(actions).item()
    
    @T.no_grad()
    def getValidationOpponentMove(self, state : T.Tensor, validMovesMask : T.Tensor, omega : float) -> int:
        validMoves = _validMovesFromMask(validMovesMask)
        if np.random.uniform(0,1) < omega:
            return np.random.choice(validMoves)

        self.model.eval()
        qvalues = self.model(state.unsqueeze(0)).squeeze()
        validqs = T.tensor([qvalues[a] for a in validMoves])
        probs = F.softmax(validqs, dim=0)
        return validMoves[T.multinomial(probs, num_samples=1)]

    def learn(self) -> None:
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

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

    @T.no_grad()
    def loadCheckpoint(self, fileName : str, learning_rate : int = None) -> None:
        cp = T.load(f'{fileName}.nn');
        self.model.load_state_dict(cp['model_state_dict']);
        self.model.optimizer.load_state_dict(cp['optimizer_state_dict']);
        self.epsilon = float(cp['epsilon'])
        if learning_rate is not None:
            for g in self.model.optimizer.param_groups:
                g['lr'] = learning_rate

    @T.no_grad()
    def saveCheckpoint(self, fileName : str) -> None:
        self.model.eval();
        try:
            T.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model.optimizer.state_dict(),
                'epsilon': str(self.epsilon),
            }, f'{fileName}.nn');

            dummy_input = createStateTensor(Connect4Board())
            T.onnx.export(self.model, dummy_input.unsqueeze(0), f"{fileName}.onnx");

            print(f"Checkpoint '{fileName}' saved.")
        except Exception as e:
            print(f"Failed to save checkpoint '{fileName}': {e}")

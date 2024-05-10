import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import random
from connect4 import Connect4Board
import connect4cnn
import matplotlib.pyplot as plt
import seaborn as sns

def log(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

def createStateTensor(board : Connect4Board) -> torch.Tensor:
    transposedBoard = torch.tensor(board._board, dtype=torch.int64).transpose(0, 1)
    onehot = F.one_hot(transposedBoard, num_classes=3).permute(2, 0, 1)
    return torch.stack([onehot]).float()

@torch.no_grad()
def getTrainingMove(qvalues : torch.Tensor, validMoves : list[int], epsilon : float) -> int:
    if len(validMoves) == 1:
        return validMoves[0]
    elif random.uniform(0,1) < epsilon:
        return random.choice(validMoves)
    
    validqs = torch.tensor([qvalues[a] for a in validMoves])
    # mean = validqs.mean()
    # std = validqs.std() + 1e-8
    # normalizedqs = (validqs - mean) / std
    # probs = F.softmax(normalizedqs, dim=0)
    probs = F.softmax(validqs, dim=0)
    return validMoves[torch.multinomial(probs, num_samples=1)]

@torch.no_grad()
def getValidationOpponentMove(model : nn.Module, board : Connect4Board, epsilon : float) -> int:
    if len(board.ValidMoves) == 1:
        return board.ValidMoves[0]
    elif random.uniform(0,1) < epsilon:
        return random.choice(board.ValidMoves)
    
    state = createStateTensor(board)    
    qvalues = model(state).squeeze()
    validqs = torch.tensor([qvalues[a] for a in board.ValidMoves])
    # mean = validqs.mean()
    # std = validqs.std() + 1e-8
    # normalizedqs = (validqs - mean) / std
    # probs = F.softmax(normalizedqs, dim=0)
    probs = F.softmax(validqs, dim=0)
    return board.ValidMoves[torch.multinomial(probs, num_samples=1)]

@torch.no_grad()
def _playValidationGame(model : nn.Module, qplayer : int, epsilon : float, qlist : list[int], maxqlist : list[int]) -> Connect4Board:
    board = Connect4Board()
    while not board.Finished:
        if qplayer == board.Player:            
            state = createStateTensor(board)
            qvalues = model(state).squeeze()
            for q in [qvalues[x] for x in board.ValidMoves]:
                qlist.append(q.item())
            action = max(board.ValidMoves, key = lambda x: qvalues[x])
            maxqlist.append(qvalues[action].item())
        else:
            action = getValidationOpponentMove(model, board, epsilon)
        board.move(action)
    return board

@torch.no_grad()
def validate(model : nn.Module, gamesPerPlayer : int, epsilon : float) -> None:
    log(f'Validation with {100*(epsilon)}% random moves')
    train = model.training
    model.eval()

    qvalues = []
    maxqvalues = []

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(model, Connect4Board.PLAYER1, epsilon, qvalues, maxqvalues)
        gk = board.gameKey
        games.add(gk)
        if board.Winner == Connect4Board.PLAYER1:
            wins += 1
        elif board.Winner == Connect4Board.PLAYER2:
            losses += 1
        else:
            draws += 1

    log(f'Player 1: {wins} won, {losses} lost, {draws} draws -> {100*wins/gamesPerPlayer:.2f}%, div: {100*len(games)/gamesPerPlayer:.2f}%')
    sns.histplot(qvalues, kde=True)
    plt.title('Distribution of Q-values')
    plt.xlabel('Q-values')
    plt.ylabel('Frequency')
    plt.show()
    qvalues = []
    sns.histplot(maxqvalues, kde=True)
    plt.title('Distribution of Max-Q-values')
    plt.xlabel('Q-values')
    plt.ylabel('Frequency')
    plt.show()
    maxqvalues = []

    wins = losses = draws = 0
    games = set()
    for _ in range(gamesPerPlayer):
        board = _playValidationGame(model, Connect4Board.PLAYER2, epsilon, qvalues, maxqvalues)
        gk = board.gameKey
        games.add(gk)
        if board.Winner == Connect4Board.PLAYER2:
            wins += 1
        elif board.Winner == Connect4Board.PLAYER1:
            losses += 1
        else:
            draws += 1

    log(f'Player 2: {wins} won, {losses} lost, {draws} draws -> {100*wins/gamesPerPlayer:.2f}%, div: {100*len(games)/gamesPerPlayer:.2f}%')
    
    sns.histplot(qvalues, kde=True)
    plt.title('Distribution of Q-values')
    plt.xlabel('Q-values')
    plt.ylabel('Frequency')
    plt.show()
    sns.histplot(maxqvalues, kde=True)
    plt.title('Distribution of Max-Q-values')
    plt.xlabel('Q-values')
    plt.ylabel('Frequency')
    plt.show()

    if train:
        model.train()

def loadCheckpoint(model : nn.Module, optimizer : torch.optim.Optimizer, fileName : str, learning_rate : int = None):
    cp = torch.load(f'{fileName}.nn');
    model.load_state_dict(cp['model_state_dict']);
    if optimizer is None:
        return
    optimizer.load_state_dict(cp['optimizer_state_dict']);
    if learning_rate is not None:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

@torch.no_grad()
def saveCheckpoint(model : nn.Module, optimizer : torch.optim.Optimizer, fileName : str):
    train = model.training
    if train: model.eval();
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{fileName}.nn');

        dummy_input = createStateTensor(Connect4Board())
        torch.onnx.export(model, dummy_input, f"{fileName}.onnx");

        log(f"Checkpoint '{fileName}' saved.")
    except Exception as e:
        log(f"Failed to save checkpoint '{fileName}': {e}")
    finally:
        if train: model.train();

def train(model : connect4cnn.Connect4Cnn, optimizer : torch.optim.Optimizer, numberOfGames : int, batch_size : int,
          epsilon : float, omega : float, 
          gamma : float = 0.9, 
          gameOffset : int = 0, log_interval : int = 5000, 
          validation_interval : int = 50000, validation_games : int = 5000,
          checkpoint_interval : int = 50000) -> None:
    losses = []
    validationLosses = []
    gameKeys = set()
    allGameKeys = set()

    training = model.training
    model.train()

    log(f"Starting training at {gameOffset} games for {numberOfGames} games with lr {optimizer.param_groups[0]['lr']} and epsilon {epsilon}.")

    qs = []
    targetqs = []
    
    targetNetwork = connect4cnn.Connect4Cnn()
    targetNetwork.load_state_dict(model.state_dict())
    targetNetwork.eval()

    for gamesPlayed in range(numberOfGames):
        env = Connect4Board()
        actions = []           

        targetq = targetNetwork(createStateTensor(env)).squeeze().clone().detach().clamp(min=-1, max=1)

        while not env.Finished:
            q = model(createStateTensor(env)).squeeze()
            qs.append(q)
            targetqs.append(targetq)
            action = getTrainingMove(q, env.ValidMoves, epsilon)            
            env.move(action)
            actions.append(action)

            if env.Finished:
                reward = 1 if env.Winner != 0 else 0.5 if env.Player == Connect4Board.PLAYER1 else 0
                targetq[action] = reward
            else:
                reward = -0.1 if env.Player == Connect4Board.PLAYER2 else 0
                nextq = targetNetwork(createStateTensor(env)).squeeze().clone().detach().clamp(min=-1,max=1)
                nextmax = -max([nextq[a] for a in env.ValidMoves]).item()
                targetq[action] = reward + gamma * nextmax
                targetq = nextq

            if len(qs) == batch_size:
                predictions = torch.stack(qs)
                targets = torch.stack(targetqs)
                loss = F.mse_loss(predictions, targets)
                losses.append(loss.item())
                validationLosses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                qs = []
                targetqs = []
                targetNetwork.load_state_dict(model.state_dict())
                targetNetwork.eval()
                if not env.Finished:
                    targetq = targetNetwork(createStateTensor(env)).squeeze().clone().detach().clamp(min=-1, max=1)

        gameKey = env.gameKey
        gameKeys.add(gameKey)
        allGameKeys.add(gameKey)    
       
        games = gameOffset + gamesPlayed + 1

        if games % log_interval == 0:
            log(f'{games}: average loss: {sum(losses)/len(losses)}, div: {100*len(gameKeys)/log_interval:.2f} ({100*len(allGameKeys)/(gamesPlayed+1):.2f})')
            losses = []
            gameKeys = set()
        if games % validation_interval == 0:
            log(f'{games} average loss over last {len(validationLosses)} games: {sum(validationLosses)/len(validationLosses)}')
            validationLosses = []
            validate(model, validation_games, omega)
        if games % checkpoint_interval == 0:
            saveCheckpoint(model, optimizer, f'connect4-{games}')

    if not training:
        model.eval()
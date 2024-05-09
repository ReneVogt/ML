import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import random
from connect4 import Connect4Board
import matplotlib.pyplot as plt
import seaborn as sns

def log(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

def createStateTensor(board : Connect4Board) -> torch.Tensor:
    state = torch.tensor(board._board, dtype=torch.float32).transpose(0, 1)
    padding = torch.zeros(1, 7)
    state = torch.cat((state, padding))
    if board.Player == Connect4Board.PLAYER2:
        state = -state
    return torch.stack([torch.stack([state])])

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
            for q in qvalues:
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

def train(model : nn.Module, optimizer : torch.optim.Optimizer, numberOfGames : int, 
          epsilon : float, omega : float, gamma : float = 0.9, 
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

    for gamesPlayed in range(numberOfGames):
        env = Connect4Board()
        stack = []
        actions = []

        while not env.Finished:
            q = model(createStateTensor(env)).squeeze()
            action = getTrainingMove(q, env.ValidMoves, epsilon)
            
            stack.append((q, env.ValidMoves.copy(), action, q.clone().detach()))
            env.move(action)
            actions.append(action)

        gameKey = env.gameKey
        gameKeys.add(gameKey)
        allGameKeys.add(gameKey)    

        (_, nextvalidmoves, action, next_q) = stack[-1]
        next_q[action] = 1 if env.Winner != Connect4Board.EMPTY else 0

        for (_, validmoves, action, targetq) in reversed(stack[:-1]):                
            next_max = -max([next_q[a] for a in nextvalidmoves]).item()
            targetq[action] = -0.1 + gamma * next_max
            next_q = targetq
            nextvalidmoves = validmoves
        
        qs = torch.stack([q for (q, _, _, _) in stack])
        targets = torch.stack([target for (_ , _, _, target) in stack])
        loss = F.mse_loss(qs, targets)
        losses.append(loss.item())
        validationLosses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
import numpy as np
import math
from board import Connect4Board

class Connect4MCTS():
    _explorationParameter = math.sqrt(2)
    def __init__(self, strength : int) -> None:
        if strength < 1:
            raise ValueError('The strength must be at least 1.')
        self._strength = strength
        self._states = {} # state -> ([count]*7, [wins]*7, validMoves)

    def getMove(self, env : Connect4Board) -> int:      
        for i in range(self._strength):
            self._evaluate(env.clone())

        (counts, _, validMoves) = self._getEntry(env)
        return max(validMoves, key=lambda x: counts[x])
    
    def _evaluate(self, env : Connect4Board) -> int:
        (counts,wins,validMoves) = self._getEntry(env)
        unplayedMoves = [a for a in validMoves if counts[a] == 0]
        if len(unplayedMoves) > 0:
            action = np.random.choice(unplayedMoves)
        else:
            parentCountLog = math.log(sum(counts))
            action = max(validMoves, key=lambda x: wins[x]/counts[x] + Connect4MCTS._explorationParameter * math.sqrt(parentCountLog/counts[x]))

        counts[action] += 1
        previousHeights = [env._getColumnHeight(a) for a in range(7)]
        previousEnv = env.clone()
        env.move(action)
        currentHeights = [env._getColumnHeight(a) for a in range(7)]
        if any(currentHeights[a] - previousHeights[a] > 1 for a in range(7)):
            print(f'INVALID FOLLOW UP STATE')
            print(f'{previousEnv.stateKey} {[previousEnv._getColumnHeight(a) for a in range(7)]} {action}')            
            print(f'{env.stateKey} {[env._getColumnHeight(a) for a in range(7)]}')
            raise MCTSException(previousEnv, env)

        if env.Finished:
            result = 1 if env.Winner != Connect4Board.EMPTY else 0.5
            wins[action] += result
            return result
        
        result = -self._evaluate(env)
        if result > 0:
            wins[action] += result

        return result
    
    def _getEntry(self, env : Connect4Board):
        entry = self._states.get(env.stateKey)
        if entry is None:
            entry = ([0] * 7, [0] * 7, [a for a in range(7) if env.is_valid(a)])
            self._states[env.stateKey] = entry
        return entry




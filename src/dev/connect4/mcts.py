import numpy as np
import math
import threading
from board import Connect4Board

class Connect4MCTS():
    _explorationParameter = math.sqrt(2)
    def __init__(self, games : int) -> None:
        if games < 1:
            raise ValueError('The number of games must be at least 1.')
        self._games = games

        # The _states dictioary maps states (Connect4Board.stateKey) to tuples
        # that contain this information:
        # - counts : list[int] 
        #       The counts how often an action was chosen.
        #       We keep track of the total count in counts[7] so we
        #       don't need to sum() every time.
        # - wins : list[int]
        #       The number of wins that were reached through each action.
        # - validMoves : list[int]
        #       The list of valid actions for the given state, so we don't need
        #       to check every time.
        # - unplayedMoves : list[int]
        #       The list of not yet played moves we need to choose randomly from,
        #       so we don't need to recollect them every time.
        self._states = {} # state -> ([count]*8, [wins]*7, validMoves, unplayedMoves)

    def getMove(self, env : Connect4Board) -> int:      
        for i in range(self._games):
            self._evaluate(env.clone())

        (counts, _, validMoves, _) = self._getEntry(env)
        return max(validMoves, key=lambda x: counts[x])
    
    def _evaluate(self, env : Connect4Board) -> int:
        (counts,wins,validMoves,unplayedMoves) = self._getEntry(env)
        if len(unplayedMoves) > 0:
            action = np.random.choice(unplayedMoves)
            unplayedMoves.remove(action)
        else:
            parentCountLog = math.log(counts[7])
            action = max(validMoves, key=lambda x: wins[x]/counts[x] + Connect4MCTS._explorationParameter * math.sqrt(parentCountLog/counts[x]))

        counts[7] += 1
        counts[action] += 1
        env.move(action)

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
            validMoves = [a for a in range(7) if env.is_valid(a)]
            entry = ([0] * 8, [0] * 7, validMoves, validMoves.copy())
            self._states[env.stateKey] = entry
        return entry




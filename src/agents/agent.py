import random
import requests
import shelve

from src.utils import Agent, Game

from src.boards.bitboard import ConnectGameBitboard


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def get_action(self, game: Game):
        return random.choice(game.get_valid_actions())


class OptimalAgent(Agent):
    """
    This agent uses the online Connect 4 solver (https://connect4.gamesolver.org/) to get the exact evaluation of each
    move.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, game: ConnectGameBitboard):
        evaluations = self.get_optimal_evaluations(game)
        # Return the index of the max evaluation, if the value is not 100 (column is full)
        valid_actions = game.get_valid_actions()
        action = max(valid_actions, key=lambda x: evaluations[x] if evaluations[x] != 100 else -100)
        return action

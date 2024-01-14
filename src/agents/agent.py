import random
import requests
import shelve

from src.utils import Agent, Game

from src.boards.bitboard import ConnectGameBitboard


class RandomAgent(Agent):
    """
    This agent chooses a random action from the list of valid actions.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, game: Game):
        return random.choice(game.get_valid_actions())

    def get_priors(self, game: Game):
        return [1/len(game.get_valid_actions()) for _ in game.get_valid_actions()]


class OptimalAgent(Agent):
    """
    This agent uses the online Connect 4 solver (https://connect4.gamesolver.org/) to get the exact evaluation of each
    move. It plays optimally, which means that always wins if it starts first and always draws if it starts second and
    the opponent plays optimally.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, game: ConnectGameBitboard):
        """
        Get the optimal action.

        Args:
            game: An object containing the game state.
        """
        evaluations = self.get_optimal_evaluations(game)
        # Return the index of the max evaluation, if the value is not 100 (column is full)
        valid_actions = game.get_valid_actions()
        random.shuffle(valid_actions)
        action = max(valid_actions, key=lambda x: evaluations[x])
        return action

    def get_priors(self, game: ConnectGameBitboard):
        """
        Get the optimal evaluations for each action.

        Args:
            game: An object containing the game state.
        """
        return self.get_optimal_evaluations(game)

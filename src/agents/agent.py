import random
from src.utils import Agent, Game


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def get_action(self, game: Game):
        return random.choice(game.get_valid_actions())

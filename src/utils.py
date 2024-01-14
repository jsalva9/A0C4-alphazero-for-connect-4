from abc import ABC, abstractmethod
from typing import Union
import requests
import os
import shelve

import yaml


class Game(ABC):
    """
    Abstract class for a game
    """

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action) -> bool:
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def check_winner(self) -> Union[int, None]:
        """
        Check if there is a winner.
        Return:
             1 if starting player wins,
             -1 if opponent wins,
             0 if there is a draw,
             None if game is not over.
        """
        pass


class Agent:
    def __init__(self):
        self._base_url = 'https://connect4.gamesolver.org/solve?pos='
        self._headers = {'User-Agent': 'Mozilla/5.0'}
        self._session = requests.Session()

        self._cache = shelve.open('../../cache/cache.db', writeback=True)
        self._cache_size = os.path.getsize('../../cache/cache.db.dat')

    @abstractmethod
    def get_action(self, game: Game):
        pass

    @abstractmethod
    def get_priors(self, game: Game):
        pass

    def get_optimal_evaluations(self, game) -> list:
        """
        Get optimal evaluations for the given game state.

        Args:
            game: Game object.

        Returns:
            List of optimal evaluations for each action.
        """
        key = "".join([str(s + 1) for s in game.history])
        if key in self._cache:
            return self._cache[key]

        url = f'{self._base_url}{key}'
        response = self._session.get(url, headers=self._headers)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}")
        scores = response.json()['score']

        if self._cache_size < 250 * 1024 * 1024:
            self._cache[key] = scores

        return scores

    def get_action_accuracy(self, game, action) -> float:
        """
        Get the accuracy of a given action.

        Args:
            game: Game object.
            action: Action to get the accuracy of.

        Returns:
            Accuracy of the given action.
        """
        evaluations = self.get_optimal_evaluations(game)
        if evaluations[action] == 100:
            return 0
        x = evaluations[action]
        while 100 in evaluations:
            evaluations.remove(100)
        if max(evaluations) == min(evaluations):
            return 1

        return (x + 22) / (max(evaluations) + 22)


class Config:
    """
    Configuration object for the project.
    """
    def __init__(self):
        # Read config from config.yaml.
        self.__root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._config_path = os.path.join(self.__root, 'config.yaml')

        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.model_dir_path = os.path.join(self.__root, self._config['model_directory'])

    def __getattr__(self, item):
        # if the item is already an attribute of self, return it
        if item in self.__dict__:
            return self.__dict__[item]

        # if the item exists as a key in self._config, return it
        try:
            return self._config[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

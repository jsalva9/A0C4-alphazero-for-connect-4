from abc import ABC, abstractmethod
from typing import Union
import requests
import os
import shelve


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

        self._cache = shelve.open('../cache/cache.db', writeback=True)
        self._cache_size = os.path.getsize('../cache/cache.db.dat')

    @abstractmethod
    def get_action(self, game: Game):
        pass

    def get_optimal_evaluations(self, game) -> list:
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
        evaluations = self.get_optimal_evaluations(game)
        if evaluations[action] == 100:
            return 0
        x = evaluations[action]
        while 100 in evaluations:
            evaluations.remove(100)
        if max(evaluations) == min(evaluations):
            return 1

        return (x + 22) / (max(evaluations) + 22)

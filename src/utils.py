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


NNConf = {
    'num_iterations': 4,
    'num_games': 30,
    'num_mcts_sims': 30,
    'c_puct': 1,
    'l2_val': 0.0001,
    'momentum': 0.9,
    'learning_rate': 0.01,
    't_policy_val': 0.0001,
    'temp_init': 1,
    'temp_final': 0.001,
    'temp_thresh': 10,
    'epochs': 10,
    'batch_size': 128,
    'dirichlet_alpha': 0.5,
    'epsilon': 0.25,
    'model_directory': "./models/",
    'num_eval_games': 12,
    'eval_win_rate': 0.55,
    'load_model': 1,
    'human_play': 0,
    'resnet_blocks': 5,
    'record_loss': 1,
    'loss_file': "loss.txt",
    'game': 2
}

from abc import ABC, abstractmethod
from typing import Union


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
        pass

    @abstractmethod
    def get_action(self, game: Game):
        pass

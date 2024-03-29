from src.boards.bitboard import ConnectGameBitboard
from src.boards.classic_board import ConnectGameClassicBoard
from src.utils import Agent

from tqdm import tqdm

import numpy as np


class TestEnvironment:
    """
    This class tests the performance of two agents against each other.

    Args:
        agent_1: An object containing the first agent.
        agent_2: An object containing the second agent.
        n_games: An integer representing the number of games to play.
        bitboard: A boolean representing whether to use bitboard or classic board.
        calculate_accuracy: A boolean representing whether to calculate the accuracy of the agents.
    """
    def __init__(self, agent_1: Agent, agent_2: Agent, n_games=1000, bitboard=True, calculate_accuracy=False):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.n_games = n_games
        self.n_wins = 0
        self.n_draws = 0

        self.bitboard = bitboard
        self.calculate_accuracy = calculate_accuracy

    def initialize_board(self):
        """
        Initialize the board.

        Returns:
            Game object.
        """
        if self.bitboard:
            return ConnectGameBitboard()
        else:
            return ConnectGameClassicBoard()

    def run(self):
        """
        Run the tests. Run half the games with agent 1 starting, half with agent 2 starting
        """
        print(f'Running first batch: {self.n_games // 2} games with agent 1 starting')
        wins_1, draws_1, move_acc_1_1, move_acc_2_1 = self.run_batch(self.n_games // 2, agent_1_starts=True)
        print(f'Running second batch: {self.n_games // 2} games with agent 2 starting')
        wins_2, draws_2, move_acc_1_2, move_acc_2_2 = self.run_batch(self.n_games // 2, agent_1_starts=False)

        self.n_wins = wins_1 + (self.n_games // 2 - wins_2 - draws_2)
        self.n_draws = draws_1 + draws_2

        print(f"\n\n -- Results --")
        win_pct = round(self.n_wins / self.n_games * 100, 2)
        print(f"Agent 1 wins {self.n_wins}/{self.n_games} ({win_pct}%) with an accuracy of {0.5 * move_acc_1_1 + 0.5 * move_acc_1_2}")
        print(f"Agent 2 wins {self.n_games - self.n_wins - self.n_draws}/{self.n_games} ({100 - win_pct}%) with an accuracy of {0.5 * move_acc_2_1 + 0.5 * move_acc_2_2}")
        print(f"Draws: {self.n_draws}\n")

    def run_batch(self, n_games, agent_1_starts=True):
        """
        Run a batch of games.

        Args:
            n_games: Number of games to play.
            agent_1_starts: A boolean representing whether agent 1 starts.

        Returns:
            Number of wins, number of draws, move accuracy of agent 1, move accuracy of agent 2.
        """
        wins = 0
        draws = 0
        x = 0 if agent_1_starts else 1

        move_acc_1 = []
        move_acc_2 = []

        for _ in tqdm(range(n_games)):
            game = self.initialize_board()
            is_over = False
            turn = 0
            while not is_over:
                if turn % 2 == x:
                    action = self.agent_1.get_action(game)
                    if self.calculate_accuracy:
                        move_acc_1.append(self.agent_1.get_action_accuracy(game, action))
                else:
                    action = self.agent_2.get_action(game)
                    if self.calculate_accuracy:
                        move_acc_2.append(self.agent_2.get_action_accuracy(game, action))
                is_over = game.step(action)
                turn += 1

            winner = game.check_winner()
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
        return wins, draws, np.mean(move_acc_1), np.mean(move_acc_2)


if __name__ == "__main__":
    from src.agents.agent import OptimalAgent, RandomAgent
    from src.agents.alpha_agent import AlphaAgent

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    agent_1 = AlphaAgent()
    agent_2 = RandomAgent()
    testing = TestEnvironment(agent_1, agent_2, n_games=50, bitboard=True, calculate_accuracy=True)
    testing.run()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

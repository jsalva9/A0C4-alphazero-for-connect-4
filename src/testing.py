from src.boards.bitboard import ConnectGameBitboard
from src.boards.classic_board import ConnectGameClassicBoard
from src.utils import Agent

import numpy as np


class TestEnvironment:
    def __init__(self, agent_1: Agent, agent_2: Agent, n_games=1000, bitboard=True, calculate_accuracy=False):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.n_games = n_games
        self.n_wins = 0
        self.n_draws = 0

        self.bitboard = bitboard

    def initialize_board(self):
        if self.bitboard:
            return ConnectGameBitboard()
        else:
            return ConnectGameClassicBoard()

    def run(self):
        # Run half the games with agent 1 starting, half with agent 2 starting
        wins_1, draws_1, move_acc_1_1, move_acc_2_1 = self.run_batch(self.n_games // 2, agent_1_starts=True)
        wins_2, draws_2, move_acc_1_2, move_acc_2_2 = self.run_batch(self.n_games // 2, agent_1_starts=False)

        self.n_wins = wins_1 + (self.n_games // 2 - wins_2 - draws_2)
        self.n_draws = draws_1 + draws_2

        print(f"Agent 1 wins {self.n_wins}/{self.n_games} with an accuracy of {0.5 * move_acc_1_1 + 0.5 * move_acc_1_2}")
        print(f"Agent 2 wins {self.n_games - self.n_wins - self.n_draws}/{self.n_games} with an accuracy of {0.5 * move_acc_2_1 + 0.5 * move_acc_2_2}")
        print(f"Draws: {self.n_draws}\n")

    def run_batch(self, n_games, agent_1_starts=True):
        wins = 0
        draws = 0
        x = 0 if agent_1_starts else 1

        move_acc_1 = []
        move_acc_2 = []

        for i in range(n_games):
            game = self.initialize_board()
            is_over = False
            turn = 0
            while not is_over:
                if turn % 2 == x:
                    action = self.agent_1.get_action(game)
                    move_acc_1.append(self.agent_1.get_action_accuracy(game, action))
                else:
                    action = self.agent_2.get_action(game)
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

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    agent_1 = RandomAgent()
    agent_2 = RandomAgent()
    testing = TestEnvironment(agent_1, agent_2, n_games=50, bitboard=True)
    testing.run()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

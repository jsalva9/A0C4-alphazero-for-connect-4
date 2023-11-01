from src.boards.bitboard import ConnectGameBitboard
from src.boards.classic_board import ConnectGameClassicBoard
from src.utils import Agent


class TestEnvironment:
    def __init__(self, agent_1: Agent, agent_2: Agent, n_games=1000, bitboard=True):
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
        wins_1, draws_1 = self.run_batch(self.n_games // 2)
        wins_2, draws_2 = self.run_batch(self.n_games // 2)

        self.n_wins = wins_1 + (self.n_games // 2 - wins_2 - draws_2)
        self.n_draws = draws_1 + draws_2

        print(f"Agent 1 wins: {self.n_wins}")
        print(f"Agent 2 wins: {self.n_games - self.n_wins - self.n_draws}")
        print(f"Draws: {self.n_draws}")

    def run_batch(self, n_games):
        wins = 0
        draws = 0

        for i in range(n_games):
            game = self.initialize_board()
            is_over = False
            turn = 0
            while not is_over:
                if turn % 2 == 0:
                    action = self.agent_1.get_action(game)
                else:
                    action = self.agent_2.get_action(game)
                is_over = game.step(action)
                turn += 1

            winner = game.check_winner()
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
        return wins, draws


if __name__ == "__main__":
    from src.agents.agent import RandomAgent

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    agent_1 = RandomAgent()
    agent_2 = RandomAgent()
    testing = TestEnvironment(agent_1, agent_2, n_games=100000, bitboard=True)
    testing.run()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

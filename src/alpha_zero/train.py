from random import shuffle

from pyinstrument import Profiler

from src.alpha_zero.neural_net import NNWrapper
from src.utils import Config
from src.alpha_zero.mcts import MonteCarloTreeSearch, TreeNode
from src.alpha_zero.eval import Evaluate
from src.boards.bitboard import ConnectGameBitboard as Game


configuration = Config()


class Train:
    """
    This class performs the training of the neural network.

    Attributes:
        game: An object containing the game state.
        net: An object containing the neural network.
        eval_net: An object containing the evaluation neural network.
    """
    def __init__(self, game: Game, net):
        self.game = game
        self.net = net
        self.eval_net = NNWrapper(game)

    def run(self):
        """
        Run the training loop.
        """
        for i in range(configuration.num_iterations):
            print("Iteration", i + 1)

            training_data = []  # list to store self play states, pis and vs

            for j in range(configuration.num_games):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game, training_data)

            # Save the current neural network model.
            self.net.save_model()

            # Load the recently saved model into the evaluator network.
            self.eval_net.load_model()

            # Train the network using self play values.
            shuffle(training_data)
            self.net.train(training_data)

            # Initialize MonteCarloTreeSearch objects for both networks.
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)

            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts)
            wins, losses = evaluator.evaluate()

            print(f'wins: {wins}, draws: {configuration.num_eval_games - wins - losses}, losses: {losses}')
            win_rate = wins / configuration.num_eval_games
            print(f'win rate: {win_rate}')

            # If the win rate is > 50%, we save it as best model. Otherwise, also save it in case we don't have one yet
            if win_rate > configuration.eval_win_rate:
                # Save current model as the best model.
                print("New model saved as best model.")
                self.net.save_model("best_model")
            else:
                print("New model discarded and previous model loaded.")
                # Discard current model and use previous best model.
                self.net.load_model()

    def play_game(self, game: Game, training_data):
        """
        Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game: An object containing the game state.
            training_data: A list to store self play states, pis and vs.
        """
        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        self_play_data = []
        turn = 0

        node = TreeNode()

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            if turn < configuration.temp_thresh:
                best_child = mcts.search(game, node, configuration.temp_init)
            else:
                best_child = mcts.search(game, node, configuration.temp_final)

            # Store state, prob and v for training.
            self_play_data.append([game.get_state_representation().copy(), best_child.parent.child_psas, 0])

            action = best_child.action
            game_over = game.step(action)  # Play the child node's action.
            turn += 1
            if game_over:
                value = game.check_winner()

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        # Update v as the value of the game result.
        for game_state in self_play_data:
            game_state[2] = -value
            training_data.append(game_state)

        # Print statistics of the MCTS
        mcts.print_stats()


if __name__ == '__main__':
    profiler = Profiler()
    profiler.start()

    game = Game()
    net = NNWrapper(game)
    train = Train(game, net)
    train.run()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
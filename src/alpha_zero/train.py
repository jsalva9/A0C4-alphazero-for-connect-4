from src.alpha_zero.neural_net import NNWrapper
from src.utils import NNConf
from src.alpha_zero.mcts import MonteCarloTreeSearch, TreeNode
from src.alpha_zero.eval import Evaluate
from src.boards.bitboard import ConnectGameBitboard as Game

import numpy as np
from copy import deepcopy


class Train:
    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.eval_net = NNWrapper(game)

    def start(self):
        """Main training loop."""
        for i in range(NNConf['num_iterations']):
            print("Iteration", i + 1)

            training_data = []  # list to store self play states, pis and vs

            for j in range(NNConf['num_games']):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game, training_data)

            # Save the current neural network model.
            self.net.save_model()

            # Load the recently saved model into the evaluator network.
            self.eval_net.load_model()

            # Train the network using self play values.
            self.net.train(training_data)

            # Initialize MonteCarloTreeSearch objects for both networks.
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)

            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts,
                                 game=self.game)
            wins, losses = evaluator.evaluate()

            print("wins:", wins)
            print("losses:", losses)

            num_games = wins + losses

            if num_games == 0:
                win_rate = 0
            else:
                win_rate = wins / num_games

            print("win rate:", win_rate)

            if win_rate > NNConf['eval_win_rate']:
                # Save current model as the best model.
                print("New model saved as best model.")
                self.net.save_model("best_model")
            else:
                print("New model discarded and previous model loaded.")
                # Discard current model and use previous best model.
                self.net.load_model()

    def play_game(self, game: Game, training_data):
        """Loop for each self-play game.

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
            if turn < NNConf['temp_thresh']:
                best_child = mcts.search(game, node, NNConf['temp_init'])
            else:
                best_child = mcts.search(game, node, NNConf['temp_final'])

            # Store state, prob and v for training.
            self_play_data.append([deepcopy(game.get_state_representation()),
                                   deepcopy(best_child.parent.child_psas),
                                   0])

            action = best_child.action
            game_over = game.step(action)  # Play the child node's action.
            turn += 1
            if game_over:
                value = game.check_winner()

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        # Update v as the value of the game result.
        for game_state in self_play_data:
            value = -value
            game_state[2] = value
            self.augment_data(game_state, training_data, game.w, game.h)

    def augment_data(self, game_state, training_data, row, column):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game_state: An object containing the state, pis and value.
            training_data: A list to store self play states, pis and vs.
            row: An integer indicating the length of the board row.
            column: An integer indicating the length of the board column.
        """
        state = deepcopy(game_state[0])
        psa_vector = deepcopy(game_state[1])

        training_data.append([state, psa_vector, game_state[2]])

        # TODO: Augment data by flipping the board state horizontally.

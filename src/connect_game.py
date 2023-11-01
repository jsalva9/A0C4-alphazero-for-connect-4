import numpy as np


class ConnectGame:
    def __init__(self):
        self.size = (6, 7)
        self.in_a_row = 4
        self.board = None
        self.turn = None  # can be 1 and -1. The next piece to be placed is going to be of sign self.turn.
        self.winner = None  # can be 1, -1, and 0 (draw). None if the game is not over.

        self.reset()

    def reset(self):
        self.board = np.zeros(self.size, dtype=int)
        self.turn = 1
        self.winner = None

    def step(self, action) -> bool:
        assert self.winner is None, "Game has already ended."
        assert self.board[action[0], action[1]] == 0, "Invalid action. The cell is already occupied."

        self.board[action[0], action[1]] = self.turn

        self.winner = self.check_winner()
        self.turn *= -1     # switch turn

        # Return True if the game is over
        return self.winner is not None

    def check_winner(self):
        # Check horizontal
        for i in range(self.size[0]):
            for j in range(self.size[1] - self.in_a_row + 1):
                if np.all(self.board[i, j:j+self.in_a_row] == self.turn):
                    return self.turn

        # Check vertical
        for i in range(self.size[0] - self.in_a_row + 1):
            for j in range(self.size[1]):
                if np.all(self.board[i:i+self.in_a_row, j] == self.turn):
                    return self.turn

        # Check diagonal
        for i in range(self.size[0] - self.in_a_row + 1):
            for j in range(self.size[1] - self.in_a_row + 1):
                if np.all(np.diag(self.board[i:i+self.in_a_row, j:j+self.in_a_row]) == self.turn):
                    return self.turn
                if np.all(np.diag(np.fliplr(self.board[i:i+self.in_a_row, j:j+self.in_a_row])) == self.turn):
                    return self.turn

        # Check draw
        if np.all(self.board != 0):
            return 0

        return None

    def get_valid_actions(self):
        # If column is not full, return last empty row
        valid_actions = []
        for j in range(self.size[1]):
            if self.board[0, j] != 0:
                continue
            for i in range(self.size[0] - 1, -1, -1):
                if self.board[i, j] == 0:
                    valid_actions.append((i, j))
                    break

        return valid_actions

    def print_board(self):
        # Print in an elegant manner using unicode characters
        print("  " + " ".join([str(i) for i in range(self.size[1])]))
        for i in range(self.size[0]):
            print(i, end=" ")
            for j in range(self.size[1]):
                if self.board[i, j] == 1:
                    print("\u25CF", end=" ")
                elif self.board[i, j] == -1:
                    print("\u25CB", end=" ")
                else:
                    print("\u25A1", end=" ")
            print()


if __name__ == '__main__':
    game = ConnectGame()
    while game.winner is None:
        game.print_board()
        print("Turn: {}".format(game.turn))
        action = tuple(map(int, input("Enter action: ").split()))
        game.step(action)
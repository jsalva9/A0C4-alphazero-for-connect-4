from src.utils import Game


class ConnectGameBitboard(Game):
    def __init__(self, width=7, height=6):
        super().__init__()
        self.w = width
        self.h = height

        self.board_state = None
        self.col_heights = None
        self.moves = None
        self.history = None
        self.node_count = None
        self.bit_shifts = None
        self.base_search_order = None

        self.reset()

    def reset(self):
        self.board_state = [0, 0]
        self.col_heights = [(self.h + 1) * i for i in range(self.w)]
        self.moves = 0
        self.history = []
        self.node_count = 0
        self.bit_shifts = self.__get_bit_shifts()
        self.base_search_order = self.__get_base_search_order()

    def __repr__(self):
        state = []
        for i in range(self.h):  # row
            row_str = str(self.h - i - 1) + " "
            for j in range(self.w):  # col
                pos = 1 << (self.h + 1) * j + i
                if self.board_state[0] & pos == pos:
                    row_str += 'x '
                elif self.board_state[1] & pos == pos:
                    row_str += 'o '
                else:
                    row_str += '. '
            state.append(row_str)
        state.append("  " + " ".join([str(i) for i in range(self.w)]))
        state.reverse()  # inverted orientation more readable
        return '\n'.join(state)

    def get_current_player(self):
        """ returns current player: 0 or 1 (0 always plays first) """
        return self.moves & 1

    def get_opponent(self):
        """ returns opponent to current player: 0 or 1 """
        return (self.moves + 1) & 1

    def get_search_order(self):
        """ returns column search order containing playable columns only """
        col_order = filter(self.can_play, self.base_search_order)
        return sorted(col_order, key=self.__col_sort, reverse=True)

    def get_mask(self):
        """ returns bitstring of all occupied positions """
        return self.board_state[0] | self.board_state[1]

    def get_key(self):
        """ returns unique game state identifier """
        return self.get_mask() + self.board_state[self.get_current_player()]

    def can_play(self, col):
        """ returns true if col (zero indexed) is playable """
        return not self.get_mask() & 1 << (self.h + 1) * col + (self.h - 1)

    def play(self, col):
        player = self.get_current_player()
        move = 1 << self.col_heights[col]
        assert self.can_play(col), f'Column {col} is full'
        self.col_heights[col] += 1
        self.board_state[player] |= move
        self.history.append(col)
        self.moves += 1

    def backtrack(self):
        opp = self.get_opponent()
        col = self.history.pop()
        self.col_heights[col] -= 1
        move = 1 << (self.col_heights[col])
        self.board_state[opp] ^= move
        self.moves -= 1

    def winning_board_state(self):
        """ returns true if last played column creates winning alignment """
        opp = self.get_opponent()
        for shift in self.bit_shifts:
            test = self.board_state[opp] & (self.board_state[opp] >> shift)
            if test & (test >> 2 * shift):
                return True
        return False if self.moves < self.w * self.h else True

    def get_score(self):
        """ returns score of complete game (evaluated for winning opponent) """
        return - (self.w * self.h + 1 - self.moves) // 2

    def __get_bit_shifts(self):
        return [
            1,  # | vertical
            self.h,  # \ diagonal
            self.h + 1,  # - horizontal
            self.h + 2  # / diagonal
        ]

    def __get_base_search_order(self):
        base_search_order = list(range(self.w))
        base_search_order.sort(key=lambda x: abs(self.w // 2 - x))
        return base_search_order

    def __col_sort(self, col):
        player = self.get_current_player()
        move = 1 << self.col_heights[col]
        count = 0
        state = self.board_state[player] | move

        for shift in self.bit_shifts:
            test = state & (state >> shift) & (state >> 2 * shift)
            if test:
                count += bin(test).count('1')

        return count

    def step(self, action) -> bool:
        self.play(action)
        if self.check_winner() is not None:
            return True
        return False

    def get_valid_actions(self):
        return [c for c in range(self.w) if self.can_play(c)]

    def check_winner(self):
        players_map = {0: 1, 1: -1}
        if self.winning_board_state():
            if self.moves == self.w * self.h:
                return 0
            return players_map[self.get_opponent()]
        return None


if __name__ == '__main__':
    game = ConnectGameBitboard()

    while game.check_winner() is None:
        print(game)
        action = int(input("Enter action: "))
        game.step(action)

    print(game)
    print(game.check_winner())

from src.utils import Game


class ConnectGameBitboard(Game):
    def __init__(self, width=7, height=6):
        super().__init__()
        self._w = width
        self._h = height

        self._board_state = None
        self._col_heights = None
        self._moves = None
        self._history = None
        self._node_count = None
        self._bit_shifts = None
        self._base_search_order = None

        self.reset()

    def reset(self):
        self._board_state = [0, 0]
        self._col_heights = [(self._h + 1) * i for i in range(self._w)]
        self._moves = 0
        self._history = []
        self._node_count = 0
        self._bit_shifts = self.__get_bit_shifts()
        self._base_search_order = self.__get_base_search_order()

    def __repr__(self):
        state = []
        for i in range(self._h):  # row
            row_str = str(self._h - i - 1) + " "
            for j in range(self._w):  # col
                pos = 1 << (self._h + 1) * j + i
                if self._board_state[0] & pos == pos:
                    row_str += 'x '
                elif self._board_state[1] & pos == pos:
                    row_str += 'o '
                else:
                    row_str += '. '
            state.append(row_str)
        state.append("  " + " ".join([str(i) for i in range(self._w)]))
        state.reverse()  # inverted orientation more readable
        return '\n'.join(state)

    def __get_current_player(self):
        """ returns current player: 0 or 1 (0 always plays first) """
        return self._moves & 1

    def __get_opponent(self):
        """ returns opponent to current player: 0 or 1 """
        return (self._moves + 1) & 1

    def __get_search_order(self):
        """ returns column search order containing playable columns only """
        col_order = filter(self.__can_play, self._base_search_order)
        return sorted(col_order, key=self.__col_sort, reverse=True)

    def __get_mask(self):
        """ returns bitstring of all occupied positions """
        return self._board_state[0] | self._board_state[1]

    def __get_key(self):
        """ returns unique game state identifier """
        return self.__get_mask() + self._board_state[self.__get_current_player()]

    def __can_play(self, col):
        """ returns true if col (zero indexed) is playable """
        return not self.__get_mask() & 1 << (self._h + 1) * col + (self._h - 1)

    def __play(self, col):
        player = self.__get_current_player()
        move = 1 << self._col_heights[col]
        assert self.__can_play(col), f'Column {col} is full'
        self._col_heights[col] += 1
        self._board_state[player] |= move
        self._history.append(col)
        self._moves += 1

    def __backtrack(self):
        opp = self.__get_opponent()
        col = self._history.pop()
        self._col_heights[col] -= 1
        move = 1 << (self._col_heights[col])
        self._board_state[opp] ^= move
        self._moves -= 1

    def __winning_board_state(self):
        """ returns true if last played column creates winning alignment """
        opp = self.__get_opponent()
        for shift in self._bit_shifts:
            test = self._board_state[opp] & (self._board_state[opp] >> shift)
            if test & (test >> 2 * shift):
                return True
        return False if self._moves < self._w * self._h else True

    def __get_score(self):
        """ returns score of complete game (evaluated for winning opponent) """
        return - (self._w * self._h + 1 - self._moves) // 2

    def __get_bit_shifts(self):
        return [
            1,  # | vertical
            self._h,  # \ diagonal
            self._h + 1,  # - horizontal
            self._h + 2  # / diagonal
        ]

    def __get_base_search_order(self):
        base_search_order = list(range(self._w))
        base_search_order.sort(key=lambda x: abs(self._w // 2 - x))
        return base_search_order

    def __col_sort(self, col):
        player = self.__get_current_player()
        move = 1 << self._col_heights[col]
        count = 0
        state = self._board_state[player] | move

        for shift in self._bit_shifts:
            test = state & (state >> shift) & (state >> 2 * shift)
            if test:
                count += bin(test).count('1')

        return count

    def step(self, action) -> bool:
        self.__play(action)
        if self.check_winner() is not None:
            return True
        return False

    def get_valid_actions(self):
        return [c for c in range(self._w) if self.__can_play(c)]

    def check_winner(self):
        players_map = {0: 1, 1: -1}
        if self.__winning_board_state():
            if self._moves == self._w * self._h:
                return 0
            return players_map[self.__get_opponent()]
        return None


if __name__ == '__main__':
    game = ConnectGameBitboard()

    while game.check_winner() is None:
        print(game)
        action = int(input("Enter action: "))
        game.step(action)

    print(game)
    print(game.check_winner())

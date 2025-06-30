from __future__ import division

import operator
from copy import deepcopy
from functools import reduce

from mcts.base.base import BaseState, BaseAction
from mcts.searcher.mcts import MCTS


class NaughtsAndCrossesState(BaseState):
    """Simple tic-tac-toe implementation used for the examples and tests."""

    def __init__(self) -> None:
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.currentPlayer = 1

    def get_current_player(self) -> int:
        return self.currentPlayer

    def get_possible_actions(self) -> list:
        possibleActions = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    possibleActions.append(Action(player=self.currentPlayer, x=i, y=j))
        return possibleActions

    def take_action(self, action: "Action") -> "NaughtsAndCrossesState":
        newState = deepcopy(self)
        newState.board[action.x][action.y] = action.player
        newState.currentPlayer = self.currentPlayer * -1
        return newState

    def is_terminal(self) -> bool:
        for row in self.board:
            if abs(sum(row)) == 3:
                return True
        for column in list(map(list, zip(*self.board))):
            if abs(sum(column)) == 3:
                return True
        for diagonal in [[self.board[i][i] for i in range(len(self.board))],
                         [self.board[i][len(self.board) - i - 1] for i in range(len(self.board))]]:
            if abs(sum(diagonal)) == 3:
                return True
        return reduce(operator.mul, sum(self.board, []), 1) != 0

    def get_reward(self) -> float:
        for row in self.board:
            if abs(sum(row)) == 3:
                return sum(row) / 3
        for column in list(map(list, zip(*self.board))):
            if abs(sum(column)) == 3:
                return sum(column) / 3
        for diagonal in [[self.board[i][i] for i in range(len(self.board))],
                         [self.board[i][len(self.board) - i - 1] for i in range(len(self.board))]]:
            if abs(sum(diagonal)) == 3:
                return sum(diagonal) / 3
        return 0


class Action(BaseAction):
    """Action representing a move in the tic-tac-toe grid."""

    def __init__(self, player: int, x: int, y: int) -> None:
        self.player = player
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return str((self.x, self.y))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.player))


if __name__ == "__main__":
    initial_state = NaughtsAndCrossesState()
    searcher = MCTS(time_limit=1000)
    action = searcher.search(initial_state=initial_state)

    print(action)

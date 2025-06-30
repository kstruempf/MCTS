from __future__ import division

import copy
import random

from mcts.base.base import BaseState, BaseAction
from mcts.searcher.mcts import MCTS


class ConnectMNKState(BaseState):
    """ConnectMNKState models a Connect(m,n,k,1,1) game that generalizes
    the famous "Connect Four" itself equal to the Connect(7,6,4,1,1) game.

    Background from wikipedia:
    Connect(m,n,k,p,q) games are another generalization of gomoku to a board
    with m×n intersections, k in a row needed to win, p stones for each player
    to place, and q stones for the first player to place for the first move
    only. Each player may play only at the lowest unoccupied place in a column.
    In particular, Connect(m,n,6,2,1) is called Connect6.
    """

    playerNames = {1: 'O', -1: 'X'}

    def __init__(self, m_columns: int = 7, n_rows: int = 6, k_connections: int = 4) -> None:
        self.mColumns = m_columns
        self.nRows = n_rows
        self.kConnections = k_connections
        self.board = [[0 for _ in range(self.mColumns)] for _ in range(self.nRows)]
        self.currentPlayer = max(ConnectMNKState.playerNames.keys())
        self.isTerminated = None
        self.reward = None
        self.possibleActions = None
        self.winingPattern = None

    def show(self) -> None:
        """Print the game board to stdout."""

        rowText = ""
        for columnIndex in range(self.mColumns):
            rowText += f" {columnIndex % 10} "
        print(rowText)

        for rowIndex in reversed(range(self.nRows)):
            rowText = ""
            for x in self.board[rowIndex]:
                if x in self.playerNames:
                    rowText += f" {self.playerNames[x]} "
                else:
                    rowText += " . "
            rowText += f" {rowIndex % 10} "
            print(rowText)

    def get_current_player(self) -> int:
        """Return the player whose turn it is."""

        return self.currentPlayer

    def get_possible_actions(self) -> list:
        """Return a (cached) list of all available actions."""

        if self.possibleActions is None:
            self.possibleActions = []
            for columnIndex in range(self.mColumns):
                for rowIndex in range(self.nRows):
                    if self.board[rowIndex][columnIndex] == 0:
                        action = Action(player=self.currentPlayer,
                                        columnIndex=columnIndex,
                                        rowIndex=rowIndex)
                        self.possibleActions.append(action)
                        break
            # Shuflle actions in order to be less predicatable when MCTS is setup with a few explorations
            # Maybe better to have it here than in the MCTS engine?
            random.shuffle(self.possibleActions)
        return self.possibleActions

    def take_action(self, action: "Action") -> "ConnectMNKState":
        """Return the successor state after applying ``action``."""

        newState = copy.copy(self)
        newState.board = copy.deepcopy(newState.board)
        newState.board[action.rowIndex][action.columnIndex] = action.player
        newState.currentPlayer = self.currentPlayer * -1
        newState.isTerminated = None
        newState.possibleActions = None
        newState.winingPattern = None
        return newState

    def is_terminal(self) -> bool:
        """Return ``True`` if the current board is a terminal state."""
        if self.isTerminated is None:
            self.isTerminated = False
            for rowIndex in range(self.nRows):
                line = self.board[rowIndex]
                lineReward = self.__getLineReward(line)
                if lineReward != 0:
                    self.isTerminated = True
                    self.reward = lineReward
                    self.winingPattern = "k-in-row"
                    break

            if not self.isTerminated:
                for columnIndex in range(self.mColumns):
                    line = []
                    for rowIndex in range(self.nRows):
                        line.append(self.board[rowIndex][columnIndex])
                    lineReward = self.__getLineReward(line)
                    if lineReward != 0:
                        self.isTerminated = True
                        self.reward = lineReward
                        self.winingPattern = "k-in-column"
                        break

            if not self.isTerminated:
                # diagonal: rowIndex = columnIndex + parameter
                for parameter in range(1 - self.mColumns, self.nRows):
                    line = []
                    for columnIndex in range(self.mColumns):
                        rowIndex = columnIndex + parameter
                        if 0 <= rowIndex < self.nRows:
                            line.append(self.board[rowIndex][columnIndex])
                    lineReward = self.__getLineReward(line)
                    if lineReward != 0:
                        self.isTerminated = True
                        self.reward = lineReward
                        self.winingPattern = "k-in-diagonal"
                        break

            if not self.isTerminated:
                # antidiagonal: rowIndex = - columnIndex + parameter
                for parameter in range(0, self.mColumns + self.nRows):
                    line = []
                    for columnIndex in range(self.mColumns):
                        rowIndex = -columnIndex + parameter
                        if 0 <= rowIndex < self.nRows:
                            line.append(self.board[rowIndex][columnIndex])
                    lineReward = self.__getLineReward(line)
                    if lineReward != 0:
                        self.isTerminated = True
                        self.reward = lineReward
                        self.winingPattern = "k-in-antidiagonal"
                        break

            if not self.isTerminated and len(self.get_possible_actions()) == 0:
                self.isTerminated = True
                self.reward = 0

        return self.isTerminated

    def __getLineReward(self, line: list) -> int:
        """Return the reward for a single line of the board."""
        lineReward = 0
        if len(line) >= self.kConnections:
            for player in ConnectMNKState.playerNames.keys():
                playerLine = [x == player for x in line]
                playerConnections = 0
                for x in playerLine:
                    if x:
                        playerConnections += 1
                        if playerConnections == self.kConnections:
                            lineReward = player
                            break
                    else:
                        playerConnections = 0
                if lineReward != 0:
                    break
        return lineReward

    def get_reward(self) -> int:
        assert self.is_terminal()
        assert self.reward is not None
        return self.reward


class Action(BaseAction):
    """Action representing a move in the Connect(m,n,k) grid."""

    def __init__(self, player: int, columnIndex: int, rowIndex: int) -> None:
        self.player = player
        self.rowIndex = rowIndex
        self.columnIndex = columnIndex

    def __str__(self) -> str:
        return str((self.columnIndex, self.rowIndex))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        return self.__class__ == (other.__class__ and
                                  self.player == other.player and
                                  self.columnIndex == other.columnIndex and
                                  self.rowIndex == other.rowIndex)

    def __hash__(self) -> int:
        return hash((self.columnIndex, self.rowIndex, self.player))


def extractStatistics(searcher, action) -> dict:
    """Return simple statistics for ``action`` from ``searcher``."""
    statistics = {}
    statistics['rootNumVisits'] = searcher.root.numVisits
    statistics['rootTotalReward'] = searcher.root.totalReward
    statistics['actionNumVisits'] = searcher.root.children[action].numVisits
    statistics['actionTotalReward'] = searcher.root.children[action].totalReward
    return statistics


def main():
    """Run a full match between two MCTS searchers, possibly with different
    parametrization, playing a Connect(m,n,k) game.

    Extraction of MCTS statistics is examplified.

    The game parameters (m,n,k) are randomly chosen.
    """

    searchers = {
        "searcher-1500ms": MCTS(time_limit=1_500),
        "searcher-1000ms": MCTS(time_limit=1_000),
        "searcher-500ms": MCTS(time_limit=500),
        "searcher-250ms": MCTS(time_limit=250)
    }

    playerNames = ConnectMNKState.playerNames

    playerSearcherNames = {}
    for player in sorted(playerNames.keys()):
        playerSearcherNames[player] = random.choice(sorted(searchers.keys()))

    runnableGames = list()
    runnableGames.append((3, 3, 3))
    runnableGames.append((7, 6, 4))
    runnableGames.append((8, 7, 5))
    runnableGames.append((9, 8, 6))
    (m, n, k) = random.choice(runnableGames)
    currentState = ConnectMNKState(m_columns=m, n_rows=n, k_connections=k)

    turn = 0
    currentState.show()
    while not currentState.is_terminal():
        turn += 1
        player = currentState.get_current_player()
        action_count = len(currentState.get_possible_actions())

        searcherName = playerSearcherNames[player]
        searcher = searchers[searcherName]

        action = searcher.search(initial_state=currentState)
        statistics = extractStatistics(searcher, action)
        currentState = currentState.take_action(action)

        print(f"at turn {turn} player {playerNames[player]}={player} ({searcherName})" +
              f" takes action (column, row)={action} amongst {action_count} possibilities")

        print("searcher statitics:" +
              f" chosen action={statistics['actionTotalReward']} total reward" +
              f" over {statistics['actionNumVisits']} visits /"
              f" all explored actions={statistics['rootTotalReward']} total reward" +
              f" over {statistics['rootNumVisits']} visits")

        print('-' * 120)
        currentState.show()

    print('-' * 120)
    if currentState.get_reward() == 0:
        print(f"Connect(m={m},n={n},k={k}) game terminates; nobody wins")
    else:
        print(f"Connect(m={m},n={n},k={k}) game terminates;" +
              f" player {playerNames[player]}={player} ({searcherName}) wins" +
              f" with pattern {currentState.winingPattern}")


if __name__ == "__main__":
    main()

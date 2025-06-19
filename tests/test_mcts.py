import os
import sys

# Allow importing the package from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcts.searcher.mcts import MCTS, random_policy
from mcts.example.naughtsandcrosses import NaughtsAndCrossesState
from mcts.example.connectmnk import ConnectMNKState


def test_mcts_returns_valid_action():
    state = NaughtsAndCrossesState()
    searcher = MCTS(iteration_limit=10)
    action = searcher.search(initial_state=state)
    assert action in state.get_possible_actions()


def test_random_policy_returns_reward():
    reward = random_policy(NaughtsAndCrossesState())
    assert reward in (-1, 0, 1)


def test_connectmnk_horizontal_win():
    state = ConnectMNKState(mColumns=3, nRows=3, kConnections=3)
    state.board[0][0] = 1
    state.board[0][1] = 1
    state.board[0][2] = 1
    assert state.is_terminal()
    assert state.get_reward() == 1


def test_naughtsandcrosses_terminal_reward_row():
    state = NaughtsAndCrossesState()
    state.board[0] = [1, 1, 1]
    assert state.is_terminal()
    assert state.get_reward() == 1

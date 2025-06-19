# MCTS

This package provides a simple way of using Monte Carlo Tree Search in any perfect information domain.

It was originally authored by [pbsinclair42](https://github.com/pbsinclair42/MCTS). This fork however complies with the
[Python Naming Convention](https://namingconvention.org/python/), provides base classes for implementing states and
actions, and includes more comprehensive examples.

## Installation

With [pip](https://pypi.org/project/monte-carlo-tree-search/): `pip install monte-carlo-tree-search`

Without pip: Download the zip/tar.gz file of the [latest release](https://github.com/kstruempf/MCTS/releases),
extract it, and run `python setup.py install`

## Quick Usage

In order to run MCTS, you must implement your own `State` class that extends `mcts.base.base.BaseState` which can fully
describe the state of the world. It must implement four methods:

- `get_current_player()`: Returns 1 if it is the maximizer player's turn to choose an action, or -1 for the minimiser
  player
- `get_possible_actions()`: Returns an iterable of all `action`s which can be taken from this state
- `take_action(action)`: Returns the state which results from taking action `action`
- `is_terminal()`: Returns `True` if this state is a terminal state
- `get_reward()`: Returns the reward for this state. Only needed for terminal states.

You must also choose a hashable representation for an action as used in `get_possible_actions` and `take_action`.
Typically, this would be a class with a custom `__hash__` method, but it could also simply be a tuple, a string, etc.
A `BaseAction` class is provided for this purpose.

Once these have been implemented, running MCTS is as simple as initializing your starting state, then running:

```python
from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS


class MyState(BaseState):
    def get_possible_actions(self) -> [any]:
        pass

    def take_action(self, action: any) -> 'BaseState':
        pass

    def is_terminal(self) -> bool:
        pass

    def get_reward(self) -> float:
        pass

    def get_current_player(self) -> int:
        pass


initial_state = MyState()

searcher = MCTS(time_limit=1000)
bestAction = searcher.search(initial_state=initial_state)
```

Here the unit of `time_limit=1000` is milliseconds. You can also use for example `iteration_limit=100` to specify the
number of rollouts. Exactly one of `time_limit` and `iteration_limit` should be specified.

```python
best_action = searcher.search(initial_state=initial_state)
print(best_action)  # the best action to take found within the time limit
```

To also receive the best reward as a return value set `need_details` to `True` in `searcher.search(...)`.

```python
best_action, reward = searcher.search(initial_state=initial_state, need_details=True)
print(best_action)  # the best action to take found within the time limit
print(reward)  # the expected reward for the best action
```

**Examples**

You can find some examples using the MCTS here:

* [naughtsandcrosses.py](https://github.com/kstruempf/MCTS/blob/main/mcts/example/naughtsandcrosses.py) is a minimal
  runnable example by [pbsinclair42](https://github.com/pbsinclair42)
* [connectmnk.py](https://github.com/kstruempf/MCTS/blob/main/mcts/example/connectmnk.py) is an example running a full
  game between two MCTS agents by [LucasBorboleta](https://github.com/LucasBorboleta)

## Collaborating

Feel free to raise a new issue for any new feature or bug you've spotted. Pull requests are also welcomed if you're
interested in directly improving the project.

### Getting Started

1. Create a virtual environment: `python -m venv .venv`
2. Activate the environment and install dependencies: `pip install -r requirements.txt`
3. Install the package in editable mode to work on it locally: `pip install -e .`
4. Run tests with `pytest` before submitting a pull request.

### Coding Guidelines

Commit message should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
This makes contributions easily comprehensible and enables us to automatically generate release notes.

Recommended tooling for developers:

* JetBrains Plugin [Conventional Commit](https://plugins.jetbrains.com/plugin/13389-conventional-commit)
  by [Edoardo Luppi](https://github.com/lppedd)
* Visual Studio
  Plugin [Conventional Commits](https://marketplace.visualstudio.com/items?itemName=vivaxy.vscode-conventional-commits)
  by [vivaxy](https://marketplace.visualstudio.com/publishers/vivaxy)

**Example commit message**

```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```

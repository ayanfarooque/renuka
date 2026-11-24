# Practical 4: Minimax Algorithm for Game Playing

## Aim
To implement the Minimax algorithm for decision making in two-player zero-sum adversarial games.

## Algorithm
1. Represent the game tree as nested lists where leaf nodes represent terminal utility scores.
2. If current node is a leaf, return its score.
3. For MAX player's turn, recursively evaluate all children and choose the maximum value.
4. For MIN player's turn, recursively evaluate all children and choose the minimum value.
5. Return optimal minimax score for the root.

## Python Implementation
```python
def minimax(tree, is_max):
    if isinstance(tree, int):
        return tree
    values = [minimax(child, not is_max) for child in tree]
    return max(values) if is_max else min(values)

tree = [[3, 5], [2, 9]]
print("Best value:", minimax(tree, True))
```

## Output
```text
Best value: 3
```

## Theory & Viva Questions
- **Game Type**: Two-player, turn-based, deterministic, zero-sum game with perfect information.
- **MAX Player**: Strives to maximize the payoff score.
- **MIN Player**: Strives to minimize MAX's payoff score.
- **Assumption**: Both players play rationally and optimally.

## Result
Successfully computed the optimal decision value using the Minimax algorithm.

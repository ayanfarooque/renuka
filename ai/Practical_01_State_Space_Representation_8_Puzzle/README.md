# Practical 1: State Space Representation and Problem Formulation --- 8 Puzzle

## Aim
To represent the 8-puzzle problem using state space formulation and generate valid successor states.

## Algorithm
1. Represent each board as a tuple; 0 is the blank tile.
2. Find the blank position using index(0).
3. Determine row and column coordinates using divmod.
4. Try its four possible moves (Right, Left, Down, Up).
5. Keep only moves that stay inside the 3x3 board boundary.
6. Swap the blank tile with the adjacent tile to generate new states.

## Python Implementation
```python
start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
goal  = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def successors(state):
    i = state.index(0)
    r, c = divmod(i, 3)
    result = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            s = list(state)
            s[i], s[j] = s[j], s[i]
            result.append(tuple(s))
    return result

print("Start:", start)
print("Goal:", goal)
print("Possible next states:")
for state in successors(start):
    print(state)
```

## Output
```text
Start: (1, 2, 3, 4, 0, 6, 7, 5, 8)
Goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)
Possible next states:
(1, 2, 3, 4, 6, 0, 7, 5, 8)
(1, 2, 3, 0, 4, 6, 7, 5, 8)
(1, 2, 3, 4, 5, 6, 7, 0, 8)
(1, 0, 3, 4, 2, 6, 7, 5, 8)
```

## Theory & Viva Questions
- **State**: A particular configuration of the problem.
- **Initial State**: The starting configuration.
- **Goal State**: The desired configuration to reach.
- **Operators**: Actions that move the system from one state to another.
- **State Space**: The set of all possible states reachable from the initial state.

## Result
Successfully formulated the 8-puzzle problem and generated valid successor states.

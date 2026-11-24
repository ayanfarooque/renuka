# Practical 4: Minimax Algorithm for Game Playing

## Aim
To implement the Minimax decision algorithm for adversarial search in a two-player, turn-based, zero-sum game tree.

---

## Comprehensive Theory

### 1. Adversarial Search & Game Environments
Adversarial search problems arise in competitive multi-agent environments where agents have conflicting utility goals. In artificial intelligence, game theory commonly focuses on games that are:
- **Two-Player:** Contested between two opponents, conventionally designated **MAX** and **MIN**.
- **Deterministic:** Transitions contain no element of chance or randomness (unlike dice games).
- **Turn-Based:** Players alternate moves sequentially.
- **Zero-Sum:** Total utility across players is constant (win for MAX is loss for MIN: $U_{MAX} + U_{MIN} = 0$).
- **Perfect Information:** Both players possess full observability of the complete game state at all times (e.g., Chess, Tic-Tac-Toe, Go).

### 2. Formal Game Formulation
A game is formally modeled as a search problem:
1. **$S_0$ (Initial State):** Starting board configuration and specification of whose turn it is.
2. **$Player(s)$:** Identifies which player (MAX or MIN) has the move in state $s$.
3. **$Actions(s)$:** Returns the set of legal moves available in state $s$.
4. **$Result(s, a)$:** The transition function specifying the state resulting from action $a$.
5. **$TerminalTest(s)$:** A boolean predicate that evaluates to true if the game has ended.
6. **$Utility(s, p)$:** Numeric payoff function for player $p$ at a terminal state $s$ (e.g., $+1$ for Win, $-1$ for Loss, $0$ for Draw).

### 3. The Minimax Decision Rule
The minimax value represents the best achievable payoff for MAX assuming MIN plays rationally to minimize MAX's payoff:
$$\text{Minimax}(s) = \begin{cases} 
\text{Utility}(s) & \text{if } TerminalTest(s) \\
\max_{a \in Actions(s)} \text{Minimax}(Result(s, a)) & \text{if } Player(s) = \text{MAX} \\
\min_{a \in Actions(s)} \text{Minimax}(Result(s, a)) & \text{if } Player(s) = \text{MIN}
\end{cases}$$

#### Step-by-Step Backpropagation
1. Traverse down to the leaf (terminal) nodes recursively using Depth-First Search.
2. Evaluate leaf utilities.
3. Propagate values upward: MIN nodes select the minimum child score; MAX nodes select the maximum child score.
4. The root node picks the action leading to the state with highest minimax value.

### 4. Complexity & Optimizations
- **Time Complexity:** $O(b^m)$, where $b$ is the legal branching factor and $m$ is maximum tree depth.
- **Space Complexity:** $O(b \cdot m)$, linear memory usage corresponding to DFS traversal.
- **Alpha-Beta Pruning:** Practical enhancement that prunes branches that cannot influence the final decision, reducing effective time complexity to $O(b^{m/2})$ without altering the final minimax value.
- **Heuristic Evaluation Functions:** In games with astronomical state spaces (e.g., Chess with $\approx 10^{40}$ states), search depth is bounded and a heuristic evaluation function estimates utility at cutoff depth.

### 5. Viva Voce Questions & Answers

**Q1: What defines a zero-sum game?**  
*Answer:* A game where the cumulative payoff of all players equals zero. Any gain in utility for one player necessitates an exact and proportional loss for the opponent.

**Q2: What assumption does the Minimax algorithm make about the opponent?**  
*Answer:* It assumes the opponent is completely rational and will always make the optimal move to minimize the maximizing player's outcome.

**Q3: How does Alpha-Beta pruning optimize Minimax without changing the result?**  
*Answer:* It maintains two values: $\alpha$ (the highest value MAX is assured of) and $\beta$ (the lowest value MIN is assured of). If a branch yields a value worse than what is already assured elsewhere, the remaining sibling nodes are skipped because they cannot affect the minimax choice.

---

## Algorithm
1. Check if the current game node is terminal (represented by an integer score). If so, return that score.
2. If `is_max` is True (MAX's turn):
   - Recursively call `minimax(child, False)` for every child node.
   - Return the maximum value among all child evaluations.
3. If `is_max` is False (MIN's turn):
   - Recursively call `minimax(child, True)` for every child node.
   - Return the minimum value among all child evaluations.
4. Output the optimal score computed for the root.

---

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

---

## Output
```text
Best value: 3
```

---

## Result
Successfully implemented the Minimax adversarial game search algorithm and determined the optimal decision value.\n
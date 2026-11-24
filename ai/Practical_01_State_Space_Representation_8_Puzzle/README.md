# Practical 1: State Space Representation and Problem Formulation --- 8 Puzzle

## Aim
To formulate the 8-puzzle problem using state space representation and generate valid successor configurations by sliding the blank tile.

---

## Comprehensive Theory

### 1. Introduction to State Space Representation
In Artificial Intelligence, problem-solving can often be formalized as searching through a space of possible problem configurations. A **state space** is a mathematical abstraction comprising the set of all valid configurations (states) that an agent or system can possibly occupy, along with the set of transitions (operators) permitted between those states.

According to Russell and Norvig, a formal problem formulation requires a **5-tuple specification**:
1. **Initial State ($S_0$):** The starting configuration of the environment from which the agent begins search (e.g., a specific scrambled tile arrangement).
2. **Actions / Operators ($A(s)$):** The legal operations executable in a given state $s$. In the 8-puzzle, operations correspond to moving the blank space (`Up`, `Down`, `Left`, `Right`).
3. **Transition Model ($Result(s, a)$):** A formal mapping returning the new state reached after executing action $a$ in state $s$.
4. **Goal Test ($GoalTest(s)$):** A boolean condition determining whether a given state matches the desired target configuration (e.g., tiles in numerical order $1$ through $8$ with the blank at the end).
5. **Path Cost Function ($c(s, a, s')$):** A function assigning numeric cost to transition from $s$ to $s'$ via $a$. For unit-cost puzzles, step cost is typically $1$.

### 2. The 8-Puzzle Domain Specifics
The 8-puzzle consists of a $3 \times 3$ grid containing 8 numbered square tiles (labeled $1$ to $8$) and one empty/blank space (conventionally denoted by $0$). A tile adjacent to the blank space can slide into the empty spot, which is conceptually equivalent to moving the blank space into an adjacent position.

#### State Space Complexity & Solvability
- **Total Permutations:** There are $9! = 362,880$ possible tile arrangements.
- **Reachable Subspaces (Parity Invariant):** The entire permutation space is partitioned into two disjoint, non-communicating subsets of equal size ($9! / 2 = 181,440$). A state is solvable if and only if its permutation parity (counted via the number of inversions) matches that of the goal state. An inversion occurs whenever a larger-numbered tile precedes a smaller-numbered tile in row-major order (ignoring the blank tile).
- **Branching Factor:**
  - Center tile: $4$ valid moves.
  - Edge tiles: $3$ valid moves.
  - Corner tiles: $2$ valid moves.
  - Average effective branching factor $b \approx 3$.

### 3. Representation Strategy in Python
- **State Structure:** A flat immutable 9-element tuple is used instead of a 2D matrix for hashability, rapid equality checks, and low memory overhead.
- **Index to Coordinate Conversion:** For any 1D index $i \in [0, 8]$, the 2D matrix coordinates $(r, c)$ are computed using integer division and modulo:
  $$\text{row } r = \lfloor i / 3 \rfloor, \quad \text{col } c = i \pmod 3$$
- **Valid Move Validation:** A directional displacement $(\Delta r, \Delta c)$ is legal if $0 \le r + \Delta r < 3$ and $0 \le c + \Delta c < 3$.

### 4. Viva Voce Questions & Answers

**Q1: What is a problem-solving agent and state space?**  
*Answer:* A goal-based agent that decides actions by formulating a sequence of transitions leading from an initial state to a goal state within a discrete representation called the state space.

**Q2: Why are exactly half of all random 8-puzzle configurations unsolvable?**  
*Answer:* Sliding a tile horizontally preserves the inversion count, while sliding a tile vertically changes the inversion count by an even number ($\pm 2$ or $0$). Therefore, the parity (odd/even) of inversions is an invariant. Any configuration with an opposite parity relative to the goal cannot reach the goal through valid moves.

**Q3: Why is tuple representation preferred over list/matrix for states?**  
*Answer:* Tuples are immutable and hashable, allowing states to be stored directly in Python sets or dictionary keys for visited-state detection in search algorithms like BFS and A*.

---

## Algorithm
1. Represent each board state as an immutable 9-element tuple where `0` denotes the blank tile.
2. Locate the blank position using `state.index(0)`.
3. Compute row `r` and column `c` coordinates using `divmod(i, 3)`.
4. Iterate over possible blank movements: Right `(0, 1)`, Left `(0, -1)`, Down `(1, 0)`, and Up `(-1, 0)`.
5. Check boundary constraints ($0 \le \text{new\_r} < 3$ and $0 \le \text{new\_c} < 3$).
6. Swap the blank tile with the target adjacent tile and append the resulting tuple to the successor list.

---

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

---

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

---

## Result
Successfully modeled the 8-puzzle problem using state space formulation and verified successor generation.\n
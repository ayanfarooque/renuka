# Practical 3: Informed Search --- Greedy Best-First Search and A*

## Aim
To implement and analyze heuristic informed search algorithms: Greedy Best-First Search and A* Search on a graph with known heuristic estimates.

---

## Comprehensive Theory

### 1. Introduction to Informed (Heuristic) Search
Informed search strategies incorporate problem-specific knowledge beyond the basic domain definition to guide the search towards goal states more efficiently. This domain knowledge is encapsulated inside a **heuristic evaluation function** $h(n)$.
- $h(n) =$ Estimated cost of the cheapest path from state $n$ to a target goal state.
- By convention, if $n$ is a goal state, $h(n) = 0$.

### 2. Greedy Best-First Search
- **Evaluation Function:** $f(n) = h(n)$.
- **Working Principle:** At every step, the algorithm expands the node that is estimated to be closest to the goal, disregarding the past cost $g(n)$ accumulated to reach that node.
- **Advantages:** Highly directed towards the goal; often very fast when heuristic estimates are accurate.
- **Shortcomings:** Can be lured into dead ends, false starts, or loops. It is **neither complete** (can loop infinitely without cycle checks) **nor optimal** (often finds longer, suboptimal solutions because it ignores path cost).
- **Complexity:** Time and space are $O(b^m)$ in the worst case, where $m$ is maximum search depth.

### 3. A* Search Algorithm
- **Evaluation Function:** $f(n) = g(n) + h(n)$
  - $g(n)$: Exact path cost accumulated from the initial state to the current node $n$.
  - $h(n)$: Estimated cost from node $n$ to the goal.
  - $f(n)$: Estimated total cost of the cheapest solution passing through node $n$.
- **Admissibility Condition:** A heuristic $h(n)$ is **admissible** if it never overestimates the actual true cost $h^*(n)$ to reach the goal:
  $$0 \le h(n) \le h^*(n) \quad \forall n$$
- **Consistency (Monotonicity) Condition:** For graph search, $h(n)$ must also be consistent:
  $$h(n) \le c(n, a, n') + h(n')$$
  This satisfies the triangle inequality and guarantees that $f(n)$ values along any path are monotonically non-decreasing.
- **Optimality:** When $h(n)$ is admissible (in tree search) or consistent (in graph search), A* is **provably complete and optimal**. It expands strictly fewer or equal nodes than any other optimal search algorithm using the same heuristic.

### 4. Comparative Analysis

| Feature | Greedy Best-First Search | A* Search |
|---|---|---|
| **Evaluation Function** | $f(n) = h(n)$ | $f(n) = g(n) + h(n)$ |
| **Path Cost Awareness** | Ignores past cost $g(n)$ | Balances past cost $g(n)$ and future estimate $h(n)$ |
| **Optimality** | Not guaranteed (often suboptimal) | Guaranteed optimal (with admissible/consistent $h$) |
| **Completeness** | Complete only with loop checks | Complete (with finite branching & positive edge costs) |
| **Degeneration** | Becomes Depth-First-like greed | When $h(n)=0$, A* degenerates into Uniform Cost Search |

### 5. Viva Voce Questions & Answers

**Q1: What does it mean for a heuristic to be admissible?**  
*Answer:* An admissible heuristic never overestimates the true remaining cost to achieve the goal; it is an optimistic assessment. Under-estimation ensures that A* will not overlook an optimal path.

**Q2: What happens if $h(n) = 0$ for all nodes in A*?**  
*Answer:* The evaluation function simplifies to $f(n) = g(n) + 0 = g(n)$. A* then behaves identically to Dijkstra's algorithm / Uniform Cost Search (UCS).

**Q3: Why did Greedy search choose path `['A', 'C', 'D']` while A* chose `['A', 'B', 'D']`?**  
*Answer:* Greedy search chose node `C` over `B` because $h(C)=1 < h(B)=2$, ignoring that reaching `C` from `A` costs $4$. A* balanced step cost ($g$) and heuristic ($h$), identifying that path $A \to B \to D$ has total cost $1 + 2 = 3$, which is cheaper than $A \to C \to D$ with cost $4 + 1 = 5$.

---

## Algorithm
- **Greedy Best-First:** Initialize priority queue with `(h[start], start, [start])`. Pop minimum $h(n)$. If node is goal, return path. Else, push unvisited neighbors keyed by $h[neighbor]$.
- **A\* Search:** Initialize priority queue with `(h[start], 0, start, [start])`. Maintain dictionary of best path costs $g$. Pop node with minimum $f(n) = g(n) + h(n)$. If goal reached, return `(path, g)`. If popped cost equals $best[node]$, relax neighbor costs with $new\_g = g + w$ and push $(new\_g + h[n], new\_g, n, path + [n])$.

---

## Python Implementation
```python
import heapq

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2)], 'C': [('D', 1)], 'D': []
}
h = {'A': 3, 'B': 2, 'C': 1, 'D': 0}

def greedy(start, goal):
    q, seen = [(h[start], start, [start])], set()
    while q:
        _, node, path = heapq.heappop(q)
        if node == goal:
            return path
        if node not in seen:
            seen.add(node)
            for n, _ in graph[node]:
                if n not in seen:
                    heapq.heappush(q, (h[n], n, path + [n]))

def astar(start, goal):
    q, best = [(h[start], 0, start, [start])], {start: 0}
    while q:
        _, cost, node, path = heapq.heappop(q)
        if node == goal:
            return path, cost
        if cost == best[node]:
            for n, w in graph[node]:
                new_cost = cost + w
                if new_cost < best.get(n, float('inf')):
                    best[n] = new_cost
                    heapq.heappush(q, (new_cost + h[n], new_cost,
                                       n, path + [n]))

print("Greedy:", greedy('A', 'D'))
print("A*:", astar('A', 'D'))
```

---

## Output
```text
Greedy: ['A', 'C', 'D']
A*: (['A', 'B', 'D'], 3)
```

---

## Result
Successfully implemented and compared Greedy Best-First Search and A* Search algorithms.\n
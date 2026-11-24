# Practical 2: Uninformed Search --- BFS, DFS and UCS

## Aim
To implement and compare three fundamental uninformed (blind) search algorithms: Breadth-First Search (BFS), Depth-First Search (DFS), and Uniform Cost Search (UCS) on a weighted directed graph.

---

## Comprehensive Theory

### 1. Concept of Uninformed Search
Uninformed search (also referred to as blind search or brute-force search) operates without any domain-specific knowledge regarding how far a given state is from the goal. These algorithms only utilize the information supplied in the problem definition: the initial state, legal operators, transition model, and goal test.

### 2. Detailed Algorithm Profiles

#### A. Breadth-First Search (BFS)
- **Core Mechanism:** Explores the state tree level by level. All shallow nodes at depth $d$ are expanded before any node at depth $d+1$.
- **Frontier Data Structure:** First-In, First-Out (FIFO) queue.
- **Completeness:** Yes, guaranteed if branching factor $b$ is finite.
- **Optimality:** Optimal if all step costs are equal (uniform step cost).
- **Time Complexity:** $O(b^d)$, where $b$ is branching factor and $d$ is depth of the shallowest goal.
- **Space Complexity:** $O(b^d)$, as every generated node must reside in memory (queue or explored set). Memory consumption is typically the primary bottleneck.

#### B. Depth-First Search (DFS)
- **Core Mechanism:** Progresses along each branch to the deepest possible level before backtracking to unexpanded branches.
- **Frontier Data Structure:** Last-In, First-Out (LIFO) stack (or system call stack via recursion).
- **Completeness:** No in infinite-depth spaces or graphs containing cycles (unless cycle checking is maintained).
- **Optimality:** Not optimal; can return a very deep or circuitous solution when a much shorter path exists.
- **Time Complexity:** $O(b^m)$, where $m$ is the maximum depth of the state space.
- **Space Complexity:** $O(b \cdot m)$, exceptionally linear in memory, storing only the active path from root to current node plus unexpanded siblings.

#### C. Uniform Cost Search (UCS)
- **Core Mechanism:** Variant of Dijkstra's algorithm. Expands the frontier node with the minimum cumulative path cost $g(n)$ from the start node.
- **Frontier Data Structure:** Priority Queue (Min-Heap keyed by $g(n)$).
- **Completeness:** Yes, provided every step cost is positive ($\ge \epsilon > 0$).
- **Optimality:** Always optimal for any non-negative edge costs.
- **Time & Space Complexity:** $O(b^{1 + \lfloor C^* / \epsilon \rfloor})$, where $C^*$ is optimal cost and $\epsilon$ is minimal positive edge weight.

### 3. Comparative Summary Matrix

| Metric | Breadth-First Search (BFS) | Depth-First Search (DFS) | Uniform Cost Search (UCS) |
|---|---|---|---|
| **Data Structure** | FIFO Queue | LIFO Stack | Min-Priority Queue (Heap) |
| **Strategy** | Level-by-level | Deepest branch first | Lowest path cost $g(n)$ first |
| **Completeness** | Yes (finite $b$) | No (in infinite/cyclic graphs) | Yes (for $\epsilon > 0$) |
| **Optimality** | Optimal for uniform costs | Suboptimal | Always optimal |
| **Time Complexity** | $O(b^d)$ | $O(b^m)$ | $O(b^{1 + \lfloor C^*/\epsilon \rfloor})$ |
| **Space Complexity** | $O(b^d)$ (Exponential) | $O(b \cdot m)$ (Linear) | $O(b^{1 + \lfloor C^*/\epsilon \rfloor})$ |

### 4. Viva Voce Questions & Answers

**Q1: Why is BFS rarely used for deep search problems?**  
*Answer:* BFS maintains all nodes at the current frontier in memory. Because space complexity is exponential ($O(b^d)$), memory is exhausted long before computational processing limits are reached.

**Q2: What is the difference between BFS and UCS?**  
*Answer:* BFS expands the node with the fewest number of steps from root, assuming identical edge weights. UCS expands the node with the lowest cumulative path cost $g(n)$, making it optimal for graphs with varying positive edge weights.

**Q3: When would you prefer DFS over BFS?**  
*Answer:* DFS is preferred when memory is constrained and solutions are dense or located deeply, or in problems like maze solving where any solution path is acceptable.

---

## Algorithm
- **BFS:** Enqueue `[start]`. While queue is not empty, pop head path. If terminal node equals `goal`, return path. Otherwise, enqueue unvisited neighbor extensions.
- **DFS:** Push `[start]` to stack. Pop top path. If goal reached, return path. Otherwise, push unvisited neighbor paths to explore deepest paths first.
- **UCS:** Push `(0, [start])` to min-heap. Pop lowest-cost path `(cost, path)`. If goal reached, return `(path, cost)`. If node cost is optimal so far, relax neighbor paths with `cost + weight` and push to heap.

---

## Python Implementation
```python
import heapq

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [], 'E': [('F', 1)], 'F': []
}

def bfs(start, goal):
    q, seen = [[start]], {start}
    while q:
        path = q.pop(0)
        if path[-1] == goal:
            return path
        for n, _ in graph[path[-1]]:
            if n not in seen:
                seen.add(n)
                q.append(path + [n])

def dfs(start, goal):
    stack, seen = [[start]], set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path
        if node not in seen:
            seen.add(node)
            for n, _ in reversed(graph[node]):
                if n not in seen:
                    stack.append(path + [n])

def ucs(start, goal):
    q, best = [(0, [start])], {start: 0}
    while q:
        cost, path = heapq.heappop(q)
        node = path[-1]
        if node == goal:
            return path, cost
        if cost == best[node]:
            for n, w in graph[node]:
                new_cost = cost + w
                if new_cost < best.get(n, float('inf')):
                    best[n] = new_cost
                    heapq.heappush(q, (new_cost, path + [n]))

print("BFS:", bfs('A', 'F'))
print("DFS:", dfs('A', 'F'))
print("UCS:", ucs('A', 'F'))
```

---

## Output
```text
BFS: ['A', 'C', 'F']
DFS: ['A', 'B', 'E', 'F']
UCS: (['A', 'C', 'F'], 7)
```
*(Note: For UCS, both paths `['A', 'C', 'F']` and `['A', 'B', 'E', 'F']` have an optimal cost of 7).*

---

## Result
Successfully implemented and evaluated uninformed search strategies (BFS, DFS, UCS).\n
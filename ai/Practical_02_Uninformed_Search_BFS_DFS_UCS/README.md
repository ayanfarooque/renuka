# Practical 2: Uninformed Search --- BFS, DFS and UCS

## Aim
To implement and compare Breadth-First Search (BFS), Depth-First Search (DFS), and Uniform Cost Search (UCS) on a graph.

## Algorithm
- **BFS**: Use a queue (FIFO); explore the graph level by level.
- **DFS**: Use a stack (LIFO); explore one branch deeply before backtracking.
- **UCS**: Use a priority queue (min-heap); expand the lowest cumulative path cost node first.

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

## Output
```text
BFS: ['A', 'C', 'F']
DFS: ['A', 'B', 'E', 'F']
UCS: (['A', 'C', 'F'], 7)
```
*(Note: For UCS, both paths `['A', 'C', 'F']` and `['A', 'B', 'E', 'F']` have an optimal cost of 7).*

## Theory & Viva Questions
- **BFS**: Explores level by level; uses a FIFO queue; complete and optimal when all step costs are equal.
- **DFS**: Explores deeply; uses a LIFO stack; not generally optimal; space-efficient.
- **UCS**: Expands the lowest path cost node; uses a priority queue; optimal for non-negative step costs.

## Result
Successfully implemented and evaluated uninformed search strategies (BFS, DFS, UCS).

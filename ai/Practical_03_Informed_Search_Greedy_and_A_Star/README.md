# Practical 3: Informed Search --- Greedy Best-First Search and A*

## Aim
To implement heuristic search algorithms: Greedy Best-First Search and A* Search.

## Algorithm
- **Greedy Best-First Search**: Choose the node with the smallest heuristic estimate h(n) to goal.
- **A* Search**: Choose the node with the smallest evaluation function f(n) = g(n) + h(n), where:
  - g(n) is the exact cost from start to node n.
  - h(n) is the estimated cost from node n to goal.

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

## Output
```text
Greedy: ['A', 'C', 'D']
A*: (['A', 'B', 'D'], 3)
```

## Theory & Viva Questions
- **Heuristic h(n)**: Estimated cost from current node n to the goal.
- **Greedy Best-First Search**: f(n) = h(n); fast and guided, but not guaranteed to be optimal.
- **A* Search**: f(n) = g(n) + h(n); optimal and complete when h(n) is admissible (never overestimates the true cost) and consistent.

## Result
Successfully implemented informed search algorithms (Greedy Best-First and A*).

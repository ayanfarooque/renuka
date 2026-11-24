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

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

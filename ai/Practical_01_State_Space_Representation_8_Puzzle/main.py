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

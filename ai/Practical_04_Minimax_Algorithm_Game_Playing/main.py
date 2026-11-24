def minimax(tree, is_max):
    if isinstance(tree, int):
        return tree
    values = [minimax(child, not is_max) for child in tree]
    return max(values) if is_max else min(values)

tree = [[3, 5], [2, 9]]
print("Best value:", minimax(tree, True))

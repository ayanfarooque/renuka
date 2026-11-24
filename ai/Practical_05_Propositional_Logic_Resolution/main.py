def resolve(c1, c2):
    for x in c1:
        if -x in c2:
            return (c1 - {x}) | (c2 - {-x})
    return None

print("Result:", resolve({1, 2}, {-1}))  # (P OR Q), (NOT P)

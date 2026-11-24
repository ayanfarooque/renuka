facts = {"human(Ram)"}
rules = {"mortal(Ram)": "human(Ram)"}

# Forward chaining
for conclusion, condition in rules.items():
    if condition in facts:
        facts.add(conclusion)
print("Forward:", sorted(facts))

# Backward chaining
def backward(goal):
    if goal in facts:
        return True
    if goal in rules:
        return backward(rules[goal])
    return False

print("Backward:", backward("mortal(Ram)"))

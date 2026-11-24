# Practical 6: First Order Logic --- Forward and Backward Chaining

## Aim
To implement Forward Chaining and Backward Chaining inference mechanisms in First Order Logic.

## Algorithm
- **Forward Chaining (Data-driven)**:
  1. Start with known facts in the knowledge base.
  2. Match antecedent conditions of inference rules.
  3. Derive new conclusions and add them to facts.
  4. Repeat until no new facts can be deduced or goal is reached.
- **Backward Chaining (Goal-driven)**:
  1. Start with the goal query.
  2. Check if goal is already a known fact; if so, return True.
  3. Look for rules concluding the goal and recursively prove their premises.
  4. Return True if all sub-goals are satisfied, else False.

## Python Implementation
```python
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
```

## Output
```text
Forward: ['human(Ram)', 'mortal(Ram)']
Backward: True
```

## Theory & Viva Questions
- **First Order Logic (FOL)**: Expresses statements using objects, predicates, functions, and quantifiers.
- **Forward Chaining**: Data-driven inference from known facts to conclusions.
- **Backward Chaining**: Goal-driven inference working backward from goal to supporting facts.

## Result
Successfully implemented forward and backward chaining for First Order Logic inference.

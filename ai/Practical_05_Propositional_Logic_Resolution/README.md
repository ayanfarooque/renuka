# Practical 5: Propositional Logic Inference Using Resolution

## Aim
To implement propositional logic inference using the resolution refutation principle.

## Algorithm
1. Represent positive literals by positive integers (e.g., P = 1, Q = 2).
2. Represent negated literals by negative integers (e.g., NOT P = -1).
3. Represent clauses as sets of literals.
4. Identify complementary literals (x and -x) across two clauses.
5. Resolve by eliminating complementary literals and taking the union of remaining literals.
6. Return the resolvent clause, or None if no resolution is possible.

## Python Implementation
```python
def resolve(c1, c2):
    for x in c1:
        if -x in c2:
            return (c1 - {x}) | (c2 - {-x})
    return None

print("Result:", resolve({1, 2}, {-1}))  # (P OR Q), (NOT P)
```

## Output
```text
Result: {2}   (Q)
```

## Theory & Viva Questions
- **Propositions**: Declarative statements that evaluate to either True or False.
- **Resolution**: An inference rule that combines two clauses containing complementary literals.
- **Resolution Refutation**: Proof technique where the negated goal is added to premises to derive a contradiction (empty clause).

## Result
Successfully performed propositional resolution to infer new logical consequences.

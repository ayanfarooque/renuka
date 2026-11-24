# Practical 6: First Order Logic --- Forward and Backward Chaining

## Aim
To implement and compare Forward Chaining (data-driven) and Backward Chaining (goal-driven) automated inference strategies in First Order Logic.

---

## Comprehensive Theory

### 1. First Order Logic (FOL) Overview
While Propositional Logic is restricted to atomic propositions and boolean connectives, **First Order Logic (Predicate Calculus)** introduces high expressive power by representing environments in terms of:
- **Objects:** Discrete entities in the domain (e.g., `Ram`, `Socrates`).
- **Predicates / Relations:** Properties of objects or relationships holding between them (e.g., `human(Ram)`, `mortal(x)`).
- **Functions:** Mappings from objects to another object.
- **Quantifiers:** Universal quantifier ($\forall x$: for all $x$) and Existential quantifier ($\exists x$: there exists an $x$).

### 2. Definite Clauses and Horn Clauses
To ensure efficient, tractable automated reasoning, inference engines frequently restrict expressions to **Horn Clauses** (specifically **Definite Clauses**):
$$\text{Definite Clause: } A_1 \land A_2 \land \dots \land A_k \to B$$
- Exactly one positive literal ($B$, called the head / conclusion).
- Zero or more body literals ($A_i$, called premises / conditions).
- A clause with no body literals is called a **Fact** (e.g., `human(Ram)`).

### 3. Forward Chaining (Data-Driven / Bottom-Up)
- **Mechanism:** Starts from the collection of known facts in the Knowledge Base (KB). Whenever all conditions of a rule match existing facts, the rule triggers and fires, asserting its conclusion into the KB.
- **Direction:** Bottom-up (facts $\to$ intermediate conclusions $\to$ goal).
- **Execution Cycle:**
  1. Match: Find all rules whose antecedent conditions are satisfied by known facts.
  2. Resolve: Select rules to fire.
  3. Act: Add derived conclusions to the knowledge base.
  4. Repeat until the query is answered or no new facts can be generated (fixed point reached).
- **Applications:** Production systems, real-time monitoring, event log analysis, synthesis tasks.

### 4. Backward Chaining (Goal-Driven / Top-Down)
- **Mechanism:** Starts from the hypothesis or query goal. It searches for rules whose conclusion matches the goal and recursively attempts to prove that all antecedent premises of those rules are true.
- **Direction:** Top-down (query goal $\to$ sub-goals $\to$ supporting facts).
- **Execution Cycle:**
  1. Check if the current goal is already known as an established fact. If yes, return `True`.
  2. Locate rules that conclude the current goal.
  3. Recursively invoke backward chaining on each prerequisite condition.
  4. If all prerequisites are substantiated, return `True`; otherwise, backtrack and try alternative rules.
- **Applications:** Automated theorem provers, logic programming languages (Prolog), diagnostic expert systems (medical, troubleshooting).

### 5. Comparative Summary

| Dimension | Forward Chaining | Backward Chaining |
|---|---|---|
| **Paradigm** | Data-driven (Bottom-up) | Goal-driven (Top-down) |
| **Starting Point** | Known facts in KB | Query/Hypothesis goal |
| **Focus** | Derives all reachable conclusions | Focuses strictly on sub-goals relevant to query |
| **Search Direction** | From premises toward goals | From goal back toward supporting facts |
| **Best Used When** | Lots of facts, goals are unknown/numerous | Specific hypothesis needs verification |

### 6. Viva Voce Questions & Answers

**Q1: What is the primary advantage of First Order Logic over Propositional Logic?**  
*Answer:* Propositional logic suffers from combinatorial explosion because separate propositions are required for every individual fact. First Order Logic allows generalization over infinite objects using variables, relations, and quantifiers.

**Q2: When is Backward Chaining preferred over Forward Chaining?**  
*Answer:* Backward chaining is preferred when you have a specific target hypothesis to prove among a vast database of facts. It ignores irrelevant facts and only explores rules pertinent to proving the query.

**Q3: What guarantees termination in chaining algorithms?**  
*Answer:* In propositional or function-free definite clause systems (Datalog), the set of all derivable ground facts is finite. In the presence of arbitrary function symbols, chaining may loop infinitely unless cycle checks and depth limits are enforced.

---

## Algorithm
- **Forward Chaining:** Iterate over rule definitions `conclusion: condition`. If `condition` exists in the set of known facts, add `conclusion` to `facts`. Repeat until convergence and display deduced facts.
- **Backward Chaining:** Define `backward(goal)`. If `goal` is already in `facts`, return `True`. If `goal` is produced by a rule, recursively call `backward` on that rule's premise condition. If all paths fail, return `False`.

---

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

---

## Output
```text
Forward: ['human(Ram)', 'mortal(Ram)']
Backward: True
```

---

## Result
Successfully implemented and evaluated Forward and Backward Chaining inference mechanisms in First Order Logic.\n
# Practical 5: Propositional Logic Inference Using Resolution

## Aim
To implement propositional logic inference using the resolution refutation principle on clauses in Conjunctive Normal Form (CNF).

---

## Comprehensive Theory

### 1. Foundations of Propositional Logic
Propositional logic is a declarative knowledge representation language based on propositions—statements that are assertively either `True` or `False`.
- **Atomic Propositions:** Primitive symbols (e.g., $P, Q, R$).
- **Logical Connectives:** Negation ($\neg$), Conjunction ($\land$), Disjunction ($\lor$), Implication ($\to$), and Biconditional ($\leftrightarrow$).
- **Entailment ($KB \models \alpha$):** Knowledge base $KB$ entails sentence $\alpha$ if and only if $\alpha$ is true in every model (world) where $KB$ is true.

### 2. Conjunctive Normal Form (CNF)
The resolution algorithm requires knowledge to be expressed in **Conjunctive Normal Form (CNF)**, which is a conjunction (AND) of one or more clauses, where each clause is a disjunction (OR) of literals:
$$\text{CNF: } (L_1 \lor L_2) \land (\neg L_2 \lor L_3) \land (\dots)$$
- **Literal:** An atomic proposition ($P$) or its negation ($\neg P$).
- **Clause:** A set or disjunction of literals (e.g., $\{P, \neg Q\} \equiv P \lor \neg Q$).

#### Conversion to CNF:
1. Eliminate biconditionals: $\alpha \leftrightarrow \beta \equiv (\alpha \to \beta) \land (\beta \to \alpha)$.
2. Eliminate implications: $\alpha \to \beta \equiv \neg \alpha \lor \beta$.
3. Apply De Morgan's laws to push negations inward: $\neg (\alpha \land \beta) \equiv \neg \alpha \lor \neg \beta$.
4. Distribute $\lor$ over $\land$: $\alpha \lor (\beta \land \gamma) \equiv (\alpha \lor \beta) \land (\alpha \lor \gamma)$.

### 3. The Resolution Rule of Inference
The resolution rule takes two clauses containing **complementary literals** (a literal and its direct negation) and combines the remaining literals into a single new resolvent clause:
$$\frac{A \lor B, \quad \neg B \lor C}{A \lor C}$$
If two clauses consist strictly of single complementary literals $B$ and $\neg B$, their resolution produces the **empty clause** (denoted by $\square$ or $\emptyset$), which represents a logical contradiction (False).

### 4. Resolution Refutation Proof
To prove whether $KB \models \alpha$:
1. Negate the query sentence: $\neg \alpha$.
2. Add $\neg \alpha$ to the knowledge base: $KB' = KB \cup \{\neg \alpha\}$.
3. Convert all sentences in $KB'$ into CNF clauses.
4. Repeatedly apply resolution to pairs of clauses with complementary literals.
5. If the empty clause $\square$ is derived, a contradiction has been established, proving $KB \models \alpha$ is **valid**. If no new clauses can be derived, $KB$ does not entail $\alpha$.
- **Refutation Completeness:** Propositional resolution is sound and refutation-complete.

### 5. Representation in Python
- Positive literal $P$ is mapped to $+1$; negated literal $\neg P$ is mapped to $-1$.
- Clauses are represented as Python `set` collections: $\{1, 2\} \equiv P \lor Q$, $\{-1\} \equiv \neg P$.
- Complementary check verifies whether $-x \in c_2$ for any literal $x \in c_1$.
- Resolvent is computed via set union and difference: $(c_1 - \{x\}) \cup (c_2 - \{-x\})$.

### 6. Viva Voce Questions & Answers

**Q1: What is a complementary literal?**  
*Answer:* A pair of literals where one is the exact negation of the other, such as $P$ and $\neg P$ (represented numerically as $1$ and $-1$).

**Q2: What is the significance of obtaining an empty clause during resolution?**  
*Answer:* The empty clause contains zero satisfiable literals and represents logical False (contradiction). In proof by refutation, obtaining an empty clause demonstrates that assuming the negation of the goal is impossible, thereby proving the original goal is true.

**Q3: Why must sentences be converted to CNF before applying resolution?**  
*Answer:* The resolution inference rule operates specifically on disjunctive clauses. CNF normalizes arbitrary nested logical formulas into a uniform AND-of-ORs structure suitable for systematic pairwise matching.

---

## Algorithm
1. Represent literals numerically: positive number for affirmative literals ($P = 1$), negative number for negated literals ($\neg P = -1$).
2. Store clauses as sets of numbers to eliminate duplicates and enable fast membership tests.
3. Search for any literal $x \in c_1$ such that its complement $-x \in c_2$.
4. If found, eliminate $x$ and $-x$ and return the union of the remaining elements: `(c1 - {x}) | (c2 - {-x})`.
5. If no complementary pairs exist, return `None`.

---

## Python Implementation
```python
def resolve(c1, c2):
    for x in c1:
        if -x in c2:
            return (c1 - {x}) | (c2 - {-x})
    return None

print("Result:", resolve({1, 2}, {-1}))  # (P OR Q), (NOT P)
```

---

## Output
```text
Result: {2}
```
*(Explanation: Resolving clause $(P \lor Q)$ with $(\neg P)$ eliminates literal $P$ and yields the resolvent clause $\{Q\}$).*

---

## Result
Successfully implemented propositional logic resolution and verified inference of logical consequences.\n
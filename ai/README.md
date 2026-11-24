# Artificial Intelligence Practicals --- Short Python Programs

All programs are self-contained, using fixed sample data and Python's standard library (no external packages required).

---

## Practicals List

| No. | Practical Name | Directory |
|---|---|---|
| 1 | State Space Representation and Problem Formulation --- 8 Puzzle | [Practical_01_State_Space_Representation_8_Puzzle](./Practical_01_State_Space_Representation_8_Puzzle/README.md) |
| 2 | Uninformed Search --- BFS, DFS and UCS | [Practical_02_Uninformed_Search_BFS_DFS_UCS](./Practical_02_Uninformed_Search_BFS_DFS_UCS/README.md) |
| 3 | Informed Search --- Greedy Best-First Search and A* | [Practical_03_Informed_Search_Greedy_and_A_Star](./Practical_03_Informed_Search_Greedy_and_A_Star/README.md) |
| 4 | Minimax Algorithm for Game Playing | [Practical_04_Minimax_Algorithm_Game_Playing](./Practical_04_Minimax_Algorithm_Game_Playing/README.md) |
| 5 | Propositional Logic Inference Using Resolution | [Practical_05_Propositional_Logic_Resolution](./Practical_05_Propositional_Logic_Resolution/README.md) |
| 6 | First Order Logic --- Forward and Backward Chaining | [Practical_06_First_Order_Logic_Chaining](./Practical_06_First_Order_Logic_Chaining/README.md) |
| 7 | Bayesian Inference --- Bayes' Rule and Bayesian Network | [Practical_07_Bayesian_Inference_Bayes_Rule](./Practical_07_Bayesian_Inference_Bayes_Rule/README.md) |
| 8 | Machine Learning --- Naive Bayes Classifier | [Practical_08_Machine_Learning_Naive_Bayes_Classifier](./Practical_08_Machine_Learning_Naive_Bayes_Classifier/README.md) |

---

## Quick Theory & Viva Revision

### 1. State Space Representation
- **State**: A particular configuration of the problem.
- **Initial State**: Starting configuration.
- **Goal State**: Desired configuration.
- **Operators**: Actions that move from one state to another.
- **State Space**: All possible states.

### 2. Uninformed Search
- **BFS**: Explores level by level; uses a FIFO queue; complete and optimal when all step costs are equal.
- **DFS**: Explores deeply; uses a LIFO stack; not generally optimal.
- **UCS**: Expands the lowest path cost; uses a priority queue; optimal for non-negative step costs.

### 3. Informed Search
- **Heuristic h(n)**: Estimated cost from node n to the goal.
- **Greedy Best-First**: f(n) = h(n); fast sometimes, not generally optimal.
- **A***: f(n) = g(n) + h(n); optimal with an admissible and consistent heuristic.

### 4. Minimax
- Used in two-player, turn-based, zero-sum games.
- MAX tries to maximize the score; MIN tries to minimize it.
- Assumes both players play optimally.

### 5. Propositional Resolution
- Propositions are statements that are true or false.
- Resolution combines clauses containing complementary literals.
- Commonly used for proof by contradiction.

### 6. First Order Logic Chaining
- FOL represents objects, properties, and relations using predicates.
- Forward chaining is data-driven (facts -> conclusions).
- Backward chaining is goal-driven (goal -> supporting facts).

### 7. Bayesian Inference
- **Prior**: Probability before observing evidence.
- **Likelihood**: Probability of evidence given a hypothesis.
- **Posterior**: Updated probability after evidence.
- **Bayes' Rule**: P(A|B) = P(B|A) * P(A) / P(B).
- **Bayesian Network**: DAG whose nodes are random variables and edges represent conditional dependencies.

### 8. Machine Learning
- **Naive Bayes** is a supervised probabilistic classifier.
- Assumes features are conditionally independent given the class.
- **Laplace smoothing** avoids zero probabilities.
- Training data contains examples with known labels; prediction assigns a class to a new example.

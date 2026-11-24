# Practical 7: Bayesian Inference --- Bayes' Rule and Bayesian Network

## Aim
To implement probabilistic reasoning and belief updating under uncertainty using Bayes' Rule and Bayesian Networks.

---

## Comprehensive Theory

### 1. Uncertainty in Artificial Intelligence
In complex, real-world environments, deterministic logic fails to handle:
- **Partial Observability:** Sensors cannot perceive all variables simultaneously.
- **Noisy Sensors:** Imperfect sensory inputs and observation errors.
- **Non-Deterministic Actions:** Actions have multiple probabilistic outcomes.

Probability theory provides the rigorous mathematical framework for quantifying degrees of belief in the presence of uncertainty.

### 2. Probability Foundations
- **Prior Probability ($P(A)$):** Unconditional probability of event $A$ prior to observing any new evidence.
- **Conditional Probability ($P(A|B)$):** Probability of event $A$ occurring given that event $B$ has already occurred:
  $$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{where } P(B) > 0$$
- **Law of Total Probability:** For a mutually exclusive and exhaustive set of hypotheses $\{A_1, A_2, \dots, A_k\}$:
  $$P(B) = \sum_{i=1}^k P(B|A_i) P(A_i)$$

### 3. Bayes' Theorem Formulation
Bayes' Rule formulates how an agent updates its belief in a hypothesis $H$ upon observing evidence $E$:
$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$
Expanding the denominator using the law of total probability:
$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E|H) \cdot P(H) + P(E|\neg H) \cdot P(\neg H)}$$

#### Anatomical Breakdown:
1. **Prior $P(H)$:** Baseline probability of hypothesis $H$ prior to evidence.
2. **Likelihood $P(E|H)$:** Probability that evidence $E$ would be observed given hypothesis $H$ is true.
3. **Marginal Evidence $P(E)$:** Total probability of observing evidence $E$ across all hypotheses (acts as a normalizing constant).
4. **Posterior $P(H|E)$:** Updated degree of belief in hypothesis $H$ after incorporating evidence $E$.

### 4. Bayesian Networks (Belief Networks)
A **Bayesian Network** is a compact probabilistic graphical model representing joint probability distributions over a set of random variables:
- **Structure:** A Directed Acyclic Graph (DAG), where:
  - **Nodes** represent random variables (discrete or continuous).
  - **Directed Edges** represent direct causal influence or conditional dependency.
- **Conditional Probability Tables (CPTs):** Each node $X_i$ is paired with a CPT specifying $P(X_i | \text{Parents}(X_i))$.
- **Full Joint Distribution Factorization (Chain Rule):**
  $$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))$$
- **Efficiency:** Drastically reduces model parameters from $O(2^n)$ in full joint distributions to polynomial size $O(n \cdot 2^k)$ (where $k$ is maximum parent count), making probabilistic inference computationally tractable.

### 5. Viva Voce Questions & Answers

**Q1: What is the fundamental difference between prior and posterior probability?**  
*Answer:* Prior probability is the initial probability assigned to a hypothesis before observing new evidence. Posterior probability is the conditional probability updated after new evidence is observed and integrated via Bayes' Rule.

**Q2: What is the purpose of the denominator in Bayes' formula?**  
*Answer:* The denominator $P(E)$ is the marginal probability of the evidence across all possible states of the world. It serves as a normalizing constant ensuring the sum of posterior probabilities over all mutually exclusive hypotheses equals $1$.

**Q3: How do Bayesian Networks avoid the curse of dimensionality?**  
*Answer:* Instead of specifying all $2^n - 1$ joint probabilities directly, Bayesian networks exploit conditional independence assertions encoded in the DAG structure, factorizing the joint distribution into local conditional probability distributions.

---

## Algorithm
1. Define the prior probability $P(\text{Rain}) = 0.2$ and conditional likelihoods:
   - $P(\text{Cloudy} | \text{Rain}) = 0.8$
   - $P(\text{Cloudy} | \neg \text{Rain}) = 0.1$
2. Compute the total marginal probability of the evidence $P(\text{Cloudy})$ using the Law of Total Probability:
   $$P(\text{Cloudy}) = P(\text{Rain}) \cdot P(\text{Cloudy}|\text{Rain}) + P(\neg \text{Rain}) \cdot P(\text{Cloudy}|\neg \text{Rain})$$
3. Compute posterior probability $P(\text{Rain} | \text{Cloudy})$ using Bayes' Rule:
   $$P(\text{Rain}|\text{Cloudy}) = \frac{P(\text{Cloudy}|\text{Rain}) \cdot P(\text{Rain})}{P(\text{Cloudy})}$$
4. Round the result to 4 decimal places and output the calculated posterior probability.

---

## Python Implementation
```python
rain = 0.2
cloudy_rain = 0.8
cloudy_no_rain = 0.1

cloudy = rain * cloudy_rain + (1 - rain) * cloudy_no_rain
result = rain * cloudy_rain / cloudy

print("P(Rain | Cloudy):", round(result, 4))
```

---

## Output
```text
P(Rain | Cloudy): 0.6667
```

---

## Result
Successfully computed posterior probability using Bayes' Rule and verified probabilistic belief updating.\n
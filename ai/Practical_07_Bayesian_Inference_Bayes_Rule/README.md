# Practical 7: Bayesian Inference --- Bayes' Rule and Bayesian Network

## Aim
To implement probabilistic reasoning using Bayes' Rule and Bayesian Networks.

## Algorithm
1. Define prior probability P(Rain) and conditional probabilities P(Cloudy | Rain) and P(Cloudy | NOT Rain).
2. Calculate marginal probability P(Cloudy) using Law of Total Probability:
   P(Cloudy) = P(Rain) * P(Cloudy | Rain) + P(NOT Rain) * P(Cloudy | NOT Rain).
3. Compute posterior probability using Bayes' Rule:
   P(Rain | Cloudy) = (P(Cloudy | Rain) * P(Rain)) / P(Cloudy).
4. Print the computed posterior probability.

## Python Implementation
```python
rain = 0.2
cloudy_rain = 0.8
cloudy_no_rain = 0.1

cloudy = rain * cloudy_rain + (1 - rain) * cloudy_no_rain
result = rain * cloudy_rain / cloudy

print("P(Rain | Cloudy):", round(result, 4))
```

## Output
```text
P(Rain | Cloudy): 0.6667
```

## Theory & Viva Questions
- **Prior Probability**: Probability of hypothesis before observing new evidence (P(H)).
- **Likelihood**: Probability of observing evidence given the hypothesis (P(E|H)).
- **Posterior Probability**: Updated probability after observing evidence (P(H|E)).
- **Bayes' Rule**: P(A|B) = P(B|A) * P(A) / P(B).
- **Bayesian Network**: A Directed Acyclic Graph (DAG) where nodes represent random variables and directed edges represent conditional dependencies.

## Result
Successfully calculated posterior probability using Bayes' Rule and Bayesian network principles.

# Practical 8: Machine Learning --- Naive Bayes Classifier

## Aim
To implement the Naive Bayes supervised classification algorithm from scratch using Laplace smoothing.

## Algorithm
1. Store labeled training samples with categorical features and class labels.
2. Calculate prior probability for each class: P(C) = count(C) / total_samples.
3. For each feature in query, calculate conditional likelihood P(x_i | C) using Laplace smoothing:
   (count(x_i, C) + 1) / (count(C) + 2).
4. Compute total class score by multiplying prior and feature likelihoods.
5. Select and predict the class with maximum score.

## Python Implementation
```python
data = [
    (["Sunny", "Hot"], "No"),
    (["Sunny", "Cool"], "No"),
    (["Cloudy", "Hot"], "Yes"),
    (["Cloudy", "Cool"], "Yes"),
    (["Cloudy", "Hot"], "Yes"),
    (["Sunny", "Hot"], "No")
]
query = ["Sunny", "Cool"]
classes = ["Yes", "No"]

def predict(query):
    scores = {}
    for c in classes:
        rows = [x for x, y in data if y == c]
        score = len(rows) / len(data)
        for i, value in enumerate(query):
            count = sum(row[i] == value for row in rows)
            score *= (count + 1) / (len(rows) + 2)
        scores[c] = score
    return max(scores, key=scores.get)

print("Prediction:", predict(query))
```

## Output
```text
Prediction: No
```

## Theory & Viva Questions
- **Naive Bayes Classifier**: A probabilistic classifier based on Bayes' Theorem with conditional independence assumption between features.
- **Conditional Independence**: Assumes each feature contributes independently to the probability of the class.
- **Laplace Smoothing**: Prevents zero-probability problem for unseen feature combinations in training data.
- **Supervised Learning**: Model learns patterns from labeled dataset to classify unseen queries.

## Result
Successfully built and evaluated a Naive Bayes Classifier with Laplace smoothing.

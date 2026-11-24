# Practical 8: Machine Learning --- Naive Bayes Classifier

## Aim
To implement the Naive Bayes supervised classification algorithm from scratch using Laplace (additive) smoothing on categorical training data.

---

## Comprehensive Theory

### 1. Introduction to Supervised Classification
Supervised classification is a machine learning paradigm where an algorithm learns a predictive mapping function $f: X \to Y$ from a labeled training dataset $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$, where $x_i = (x_{i1}, \dots, x_{id})$ is a feature vector and $y_i \in \{C_1, \dots, C_K\}$ is the target discrete class label.

- **Generative vs. Discriminative Models:**
  - Discriminative models (e.g., Logistic Regression, SVM) model the decision boundary directly via $P(Y|X)$.
  - Generative models (e.g., Naive Bayes) model the joint distribution $P(X, Y) = P(X|Y)P(Y)$ and apply Bayes' theorem to compute the posterior $P(Y|X)$.

### 2. Mathematical Formulation of Naive Bayes
According to Bayes' Theorem, the posterior probability of class $C$ given feature vector $X = (x_1, x_2, \dots, x_d)$ is:
$$P(C | X) = \frac{P(X | C) \cdot P(C)}{P(X)}$$
Since $P(X)$ is identical for all candidate classes, the classification rule maximizes the numerator via the **Maximum A Posteriori (MAP)** decision:
$$\hat{y} = \arg\max_{c \in C} P(C=c) \cdot P(x_1, x_2, \dots, x_d | C=c)$$

### 3. The "Naive" Conditional Independence Assumption
To compute the joint likelihood $P(x_1, \dots, x_d | C)$ directly, one would require an exponential number of training examples ($O(|V|^d)$). The Naive Bayes classifier overcomes this by assuming that **all features are conditionally independent given the class label**:
$$P(x_1, x_2, \dots, x_d | C=c) = \prod_{i=1}^d P(x_i | C=c)$$
Thus, the final classification decision simplifies to:
$$\hat{y} = \arg\max_{c \in C} \left[ P(C=c) \prod_{i=1}^d P(x_i | C=c) \right]$$

### 4. Zero-Frequency Problem & Laplace (Additive) Smoothing
If an unseen instance presents a feature value $x_i$ that never occurred with class $c$ in the training dataset, the maximum likelihood estimate gives $P(x_i | c) = 0$. Because probabilities are multiplied together, a single zero probability forces the entire posterior product to zero:
$$P(C=c) \cdot \dots \cdot 0 \cdot \dots = 0$$
To prevent this pathological failure, **Laplace (Add-1) Smoothing** is applied:
$$P(x_i = v | C=c) = \frac{\text{count}(x_i = v, c) + 1}{\text{count}(c) + K}$$
where:
- $\text{count}(x_i = v, c)$ is the frequency of feature value $v$ occurring in class $c$.
- $\text{count}(c)$ is the total count of instances labeled with class $c$.
- $K$ is the number of distinct values the feature can assume (or 2 for binary attributes).

### 5. Strengths, Weaknesses, and Applications
- **Strengths:**
  - Computationally fast for both training and inference ($O(N \cdot d)$).
  - Performs surprisingly well even when the independence assumption is violated.
  - Robust to irrelevant features and handles high-dimensional data effectively.
- **Weaknesses:**
  - Independent feature assumption rarely holds strictly in natural domains.
  - Probability estimates themselves can be poorly calibrated even if the predicted class label is correct.
- **Applications:** Spam email filtering, sentiment classification, news categorization, medical risk prediction.

### 6. Viva Voce Questions & Answers

**Q1: Why is this algorithm described as "Naive"?**  
*Answer:* It is called "naive" because it makes the extremely strong and unrealistic assumption that all input attributes are completely conditionally independent of one another given the class label.

**Q2: What is Laplace smoothing and why is it essential?**  
*Answer:* Laplace smoothing adds a small pseudo-count (typically 1) to both numerator and denominator when calculating feature likelihoods. It prevents any zero probability from nullifying the entire product during classification of unobserved feature values.

**Q3: Can Naive Bayes handle continuous numerical features?**  
*Answer:* Yes, through Gaussian Naive Bayes, which assumes continuous attributes follow a Gaussian (normal) distribution and calculates likelihoods using the probability density function parameterized by sample mean $\mu$ and variance $\sigma^2$.

---

## Algorithm
1. Store labeled training instances consisting of feature tuples and corresponding class labels.
2. Calculate prior probabilities for each class:
   $$P(C) = \frac{\text{total examples in class } C}{\text{total training examples}}$$
3. For each feature in query vector $X$:
   - Count matching occurrences where $x_i = \text{query}[i]$ within class $C$.
   - Calculate smoothed conditional likelihood using Laplace formula:
     $$P(x_i | C) = \frac{\text{count}(x_i, C) + 1}{\text{len}(C) + 2}$$
4. Multiply class prior $P(C)$ with each feature's conditional likelihood to obtain total score for class $C$.
5. Select and return the class label having the maximum posterior score.

---

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

---

## Output
```text
Prediction: No
```

---

## Result
Successfully implemented the Naive Bayes Classifier from scratch with Laplace smoothing and verified query classification.\n
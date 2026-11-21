# Recommender Systems Practicals

**Language:** Python  
**Libraries:** NumPy, Pandas, Scikit-learn, SciPy

### Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy
```

---

## Practical 1: Popularity-Based Recommendation (Top-N)

### Aim
To recommend the Top-N most popular items based on user interactions.

### Code

```python
import pandas as pd

# Step 1: Create dataset
items = ['A', 'A', 'B', 'C', 'B', 'A']
df = pd.DataFrame({'Item': items})

# Step 2: Calculate popularity
popularity = df['Item'].value_counts()

# Step 3: Get Top-2 items
top = popularity.head(2)

# Step 4: Display recommendations
print("Top Recommendations:\n", top)
```

### Algorithm
1. Start.
2. Import Pandas.
3. Create a dataset of item interactions.
4. Count item occurrences using `value_counts()`.
5. Sort items by popularity.
6. Select the Top-N items.
7. Display recommendations.
8. Stop.

### Result
Successfully implemented a popularity-based recommendation system.

---

## Practical 2: Similarity Measures (Cosine, Pearson, Jaccard)

### Aim
To calculate similarity between two vectors using Cosine, Pearson, and Jaccard measures.

### Code

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define vectors
A, B = [1, 2, 3], [2, 4, 6]
X, Y = {1, 2, 3}, {2, 3, 4}

# Step 2: Calculate Cosine Similarity
cos = cosine_similarity([A], [B])[0][0]

# Step 3: Calculate Pearson Correlation
pearson = np.corrcoef(A, B)[0][1]

# Step 4: Calculate Jaccard Similarity
jaccard = len(X & Y) / len(X | Y)

# Step 5: Display results
print("Cosine:", cos)
print("Pearson:", pearson)
print("Jaccard:", jaccard)
```

### Algorithm
1. Start.
2. Import NumPy and cosine similarity.
3. Define two numerical vectors and two sets.
4. Calculate cosine similarity.
5. Calculate Pearson correlation.
6. Find intersection and union of sets.
7. Calculate Jaccard similarity.
8. Display all similarity scores.
9. Stop.

### Result
Successfully implemented Cosine Similarity, Pearson Correlation, and Jaccard Similarity.

---

## Practical 3: Content-Based Filtering Using TF-IDF

### Aim
To recommend similar movies using TF-IDF and Cosine Similarity.

### Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create movie descriptions
movies = [
    "action adventure",
    "action superhero",
    "romantic drama",
    "superhero fantasy"
]

# Step 2: Convert text into TF-IDF vectors
tfidf = TfidfVectorizer()
M = tfidf.fit_transform(movies)

# Step 3: Calculate similarity
sim = cosine_similarity(M)

# Step 4: Find similar movies to movie 0
ranking = sim[0].argsort()[::-1][1:3]

# Step 5: Display recommendations
print("Recommended Movies:", ranking)
```

### Algorithm
1. Start.
2. Import TF-IDF and cosine similarity libraries.
3. Create a list of movie descriptions.
4. Convert text into numerical TF-IDF vectors.
5. Calculate cosine similarity between movies.
6. Select the similarity scores of the target movie.
7. Sort scores in descending order.
8. Exclude the movie itself.
9. Display Top-N recommendations.
10. Stop.

### Result
Successfully implemented content-based filtering using TF-IDF and cosine similarity.

---

## Practical 4: User Profile Learning and Item Ranking

### Aim
To create a user profile and rank items according to user preferences.

### Code

```python
import numpy as np

# Step 1: Define item feature vectors
items = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1]
])

# Step 2: Create user profile from liked items
profile = items[:2].mean(axis=0)

# Step 3: Calculate item similarity scores
scores = items @ profile / (
    np.linalg.norm(items, axis=1) *
    np.linalg.norm(profile)
)

# Step 4: Rank items
ranking = scores.argsort()[::-1]

# Step 5: Display results
print("User Profile:", profile)
print("Item Scores:", scores)
print("Ranking:", ranking)
```

### Algorithm
1. Start.
2. Import NumPy.
3. Define item feature vectors.
4. Select items liked by the user.
5. Calculate the average feature vector.
6. Create the user profile.
7. Calculate cosine similarity between the profile and items.
8. Sort items by similarity scores.
9. Display the ranking.
10. Stop.

### Result
Successfully implemented user profile learning and item ranking using feature vectors.

---

## Practical 5: User-Based Collaborative Filtering

### Aim
To recommend items using similarity between users.

### Code

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create user-item rating matrix
R = np.array([
    [5, 4, 0],
    [5, 4, 5],
    [1, 2, 1]
])

# Step 2: Calculate user similarity
S = cosine_similarity(R)

# Step 3: Select target user U1
weights = S[0, 1:]

# Step 4: Predict rating for missing item C
ratings = R[1:, 2]
prediction = weights @ ratings / np.abs(weights).sum()

# Step 5: Display results
print("User Similarity:\n", S)
print("Predicted Rating:", prediction)
```

### Algorithm
1. Start.
2. Import NumPy and cosine similarity.
3. Create a user-item rating matrix.
4. Calculate similarity between users.
5. Select the target user.
6. Identify similar users.
7. Collect ratings of similar users for the missing item.
8. Calculate the weighted predicted rating.
9. Display the prediction.
10. Stop.

### Result
Successfully implemented User-Based Collaborative Filtering using cosine similarity.

---

## Practical 6: Item-Based Collaborative Filtering

### Aim
To recommend items based on similarity between items.

### Code

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create user-item rating matrix
R = np.array([
    [5, 4, 0],
    [4, 5, 0],
    [0, 1, 5]
])

# Step 2: Calculate item-item similarity
S = cosine_similarity(R.T)

# Step 3: Select target user
user = R[0]
rated = user > 0

# Step 4: Predict scores for all items
scores = S[:, rated] @ user[rated] / np.abs(S[:, rated]).sum(axis=1)

# Step 5: Rank items
ranking = scores.argsort()[::-1]

# Step 6: Display results
print("Item Similarity:\n", S)
print("Item Scores:", scores)
print("Ranking:", ranking)
```

### Algorithm
1. Start.
2. Import NumPy and cosine similarity.
3. Create a user-item rating matrix.
4. Transpose the matrix using `R.T`.
5. Calculate item-item similarity.
6. Select the target user's ratings.
7. Calculate similarity-weighted item scores.
8. Rank items in descending order.
9. Display recommendations.
10. Stop.

### Result
Successfully implemented Item-Based Collaborative Filtering using cosine similarity.

---

## Practical 7: Matrix Factorization Using SVD

### Aim
To decompose a user-item rating matrix using Singular Value Decomposition.

### Code

```python
import numpy as np

# Step 1: Create rating matrix
R = np.array([
    [5, 4, 0],
    [4, 0, 5],
    [0, 5, 4]
])

# Step 2: Apply SVD
U, S, Vt = np.linalg.svd(R)

# Step 3: Select latent factors
k = 2

# Step 4: Reconstruct approximate matrix
pred = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Step 5: Display predicted matrix
print("Predicted Matrix:\n", np.round(pred, 2))
```

### Algorithm
1. Start.
2. Import NumPy.
3. Create a user-item rating matrix.
4. Apply Singular Value Decomposition.
5. Obtain U, S, and Vt matrices.
6. Select the number of latent factors.
7. Reconstruct the approximate rating matrix.
8. Display the predicted matrix.
9. Stop.

### Result
Successfully implemented matrix factorization using SVD.

**Note:** This is a basic SVD demonstration. It treats zero entries as actual zeros, not missing ratings. Production recommendation systems use specialized methods to handle missing interactions.

---

## Practical 8: Evaluation Using RMSE, Precision@K and Recall@K

### Aim
To evaluate a recommender system using RMSE, Precision@K, and Recall@K.

### Code

```python
import numpy as np

# Step 1: Define actual and predicted ratings
actual = np.array([5, 3, 4, 2])
predicted = np.array([4, 3, 5, 2])

# Step 2: Calculate RMSE
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

# Step 3: Define recommended and relevant items
recommended = {'A', 'B', 'C', 'D'}
relevant = {'A', 'C', 'E'}

# Step 4: Calculate common relevant items
hits = len(recommended & relevant)

# Step 5: Calculate Precision and Recall
precision = hits / len(recommended)
recall = hits / len(relevant)

# Step 6: Display results
print("RMSE:", rmse)
print("Precision:", precision)
print("Recall:", recall)
```

### Algorithm
1. Start.
2. Import NumPy.
3. Define actual and predicted ratings.
4. Calculate squared prediction errors.
5. Calculate RMSE.
6. Define recommended and relevant items.
7. Find common relevant items.
8. Calculate Precision@K.
9. Calculate Recall@K.
10. Display evaluation metrics.
11. Stop.

### Result
Successfully evaluated the recommender system using RMSE, Precision, and Recall.

---

## Practical 9: Hybrid Recommender System

### Aim
To combine content-based and collaborative filtering techniques using a weighted hybrid approach.

### Code

```python
import numpy as np

# Step 1: Define content-based scores
CB = np.array([0.9, 0.7, 0.4])

# Step 2: Define collaborative filtering scores
CF = np.array([0.6, 0.9, 0.8])

# Step 3: Set weight
alpha = 0.7

# Step 4: Calculate hybrid scores
scores = alpha * CB + (1-alpha) * CF

# Step 5: Rank items
ranking = scores.argsort()[::-1]

# Step 6: Display results
print("Hybrid Scores:", scores)
print("Ranking:", ranking)
```

### Algorithm
1. Start.
2. Import NumPy.
3. Define content-based scores.
4. Define collaborative filtering scores.
5. Set the weight alpha.
6. Combine both scores using the weighted formula.
7. Rank items in descending order.
8. Display recommendations.
9. Stop.

### Result
Successfully implemented a hybrid recommender system by combining content-based and collaborative filtering scores.

---

## Practical 10: Industrial Recommender Systems

### Aim
To study industrial recommender systems used by Netflix, Amazon, and YouTube.

### Theory

Industrial recommender systems use user behavior, interaction history, and machine learning techniques to provide personalized recommendations.

### 1. Netflix

- Recommends movies and TV shows.
- Uses viewing history, ratings, and preferences.
- Uses personalized recommendation and ranking techniques.

**Algorithm:**
1. Collect viewing history.
2. Analyze user preferences.
3. Identify relevant content.
4. Rank movies and shows.
5. Display recommendations.

### 2. Amazon

- Recommends products.
- Uses purchase history, browsing, and ratings.
- Uses item-based collaborative filtering and personalized ranking.

**Algorithm:**
1. Collect purchase and browsing data.
2. Identify similar products.
3. Calculate product relevance.
4. Rank products.
5. Display recommendations.

### 3. YouTube

- Recommends videos.
- Uses watch history, search history, and engagement.
- Uses candidate generation and ranking.

**Algorithm:**
1. Collect user interaction data.
2. Generate candidate videos.
3. Calculate relevance scores.
4. Rank videos.
5. Display personalized recommendations.

### Comparison

| Platform | Main Recommendation | Data Used |
|---|---|---|
| Netflix | Hybrid Recommendation | Watch history |
| Amazon | Item-Based CF | Purchase history |
| YouTube | Candidate Generation and Ranking | Watch time |

### Result
Successfully studied and compared industrial recommender systems used by Netflix, Amazon, and YouTube.

---

## Quick Revision Table

| No. | Practical | Main Concept |
|---|---|---|
| 1 | Popularity-Based | `value_counts()` |
| 2 | Similarity Measures | Cosine, Pearson, Jaccard |
| 3 | Content-Based Filtering | TF-IDF |
| 4 | User Profile Learning | Feature Vectors |
| 5 | User-Based CF | User Similarity |
| 6 | Item-Based CF | Item Similarity |
| 7 | Matrix Factorization | SVD |
| 8 | Evaluation | RMSE, Precision, Recall |
| 9 | Hybrid System | Weighted Scores |
| 10 | Case Study | Netflix, Amazon, YouTube |

---

## Important Formulas

**Cosine Similarity**

Cos(A,B) = (A · B) / (||A|| × ||B||)

**Pearson Correlation**

r = Σ(x - x̄)(y - ȳ) / √[Σ(x - x̄)² × Σ(y - ȳ)²]

**Jaccard Similarity**

J(A,B) = |A ∩ B| / |A ∪ B|

**TF-IDF**

TF-IDF = TF × IDF

**User Profile**

P(u) = Average of liked item feature vectors

**Hybrid Recommendation**

Score = α × CB + (1 - α) × CF

**RMSE**

RMSE = √[Σ(actual - predicted)² / n]

**Precision@K**

Precision = Relevant items in Top-K / K

**Recall@K**

Recall = Relevant items in Top-K / Total relevant items

---

**Note:** All practical programs use predefined sample data. No external datasets, CSV files, or input/output files are required. These are short educational implementations intended for practical examinations.

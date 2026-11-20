
Recommender Systems Practicals

Language: Python
Libraries: NumPy, Pandas, Scikit-learn, SciPy

Install dependencies:

pip install numpy pandas scikit-learn scipy

---

1. Popularity-Based Recommendation (Top-N)

import pandas as pd

items = ['A','A','B','C','B','A']
print(pd.Series(items).value_counts().head(2))

2. Similarity Measures (Cosine, Pearson, Jaccard)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

A, B = [1,2,3], [2,4,6]
X, Y = {1,2,3}, {2,3,4}

print("Cosine:", cosine_similarity([A],[B])[0][0])
print("Pearson:", np.corrcoef(A,B)[0][1])
print("Jaccard:", len(X&Y)/len(X|Y))

3. Content-Based Filtering Using TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = ["action adventure", "action superhero",
          "romantic drama", "superhero fantasy"]

M = TfidfVectorizer().fit_transform(movies)
S = cosine_similarity(M)

print("Recommendations:", S[0].argsort()[-2:][::-1])

4. User Profile Learning and Item Ranking

import numpy as np

items = np.array([[1,0,1],[1,0,1],[0,1,0],[0,1,1]])

profile = items[:2].mean(axis=0)
scores = items @ profile / (
    np.linalg.norm(items,axis=1) * np.linalg.norm(profile)
)

print("Profile:", profile)
print("Ranking:", scores.argsort()[::-1])

5. User-Based Collaborative Filtering

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

R = np.array([[5,4,0],[5,4,5],[1,2,1]])

S = cosine_similarity(R)
scores = S[0,1:] @ R[1:,2] / np.abs(S[0,1:]).sum()

print("Similarities:", S[0])
print("Predicted Rating:", scores)

6. Item-Based Collaborative Filtering

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

R = np.array([[5,4,0],[4,5,0],[0,1,5]])

S = cosine_similarity(R.T)
user = R[0]
rated = user > 0

scores = S[:,rated] @ user[rated] / np.abs(S[:,rated]).sum(axis=1)

print("Item Similarity:\n", S)
print("Ranking:", scores.argsort()[::-1])

7. Matrix Factorization Using SVD

import numpy as np

R = np.array([[5,4,0],[4,0,5],[0,5,4]])

U, S, V = np.linalg.svd(R)
k = 2

pred = U[:,:k] @ np.diag(S[:k]) @ V[:k,:]

print(np.round(pred,2))

8. Evaluation Using RMSE, Precision@K and Recall@K

import numpy as np

actual = np.array([5,3,4,2])
predicted = np.array([4,3,5,2])

rmse = np.sqrt(np.mean((actual-predicted)**2))

recommended = {'A','B','C','D'}
relevant = {'A','C','E'}

hits = len(recommended & relevant)

print("RMSE:", rmse)
print("Precision:", hits/len(recommended))
print("Recall:", hits/len(relevant))

9. Hybrid Recommender System

import numpy as np

CB = np.array([0.9,0.7,0.4])
CF = np.array([0.6,0.9,0.8])

alpha = 0.7
scores = alpha*CB + (1-alpha)*CF

print("Hybrid Scores:", scores)
print("Ranking:", scores.argsort()[::-1])

10. Industrial Recommender Systems — Case Study

This practical is theory-based. No Python implementation is necessary.

Platform| Recommendation Technique| Data Used
Netflix| Hybrid Filtering| Watch history, ratings
Amazon| Item-Based Collaborative Filtering| Purchases, browsing
YouTube| Candidate Generation and Ranking| Watch time, search history

Working

Netflix:

1. Collect viewing history.
2. Analyze user preferences.
3. Find relevant movies.
4. Rank and recommend content.

Amazon:

1. Collect browsing and purchase data.
2. Identify similar products.
3. Calculate product relevance.
4. Recommend products.

YouTube:

1. Candidate Generation: Select potentially relevant videos.
2. Ranking: Rank videos according to predicted relevance.
3. Display personalized recommendations.

---

Quick Revision

Practical| Main Function / Concept
1| "value_counts()"
2| Cosine, Pearson, Jaccard
3| "TfidfVectorizer()"
4| Average feature vectors
5| User-user similarity
6| "cosine_similarity(R.T)"
7| "np.linalg.svd()"
8| RMSE, Precision, Recall
9| Weighted score combination
10| Netflix, Amazon, YouTube

Note: All programs use small, predefined datasets and print results directly. Practical 5 and 6 use simplified similarity-weighted predictions. Practical 7 demonstrates basic SVD reconstruction, treating zeros as observed values. These are educational implementations, not production-grade recommendation systems.
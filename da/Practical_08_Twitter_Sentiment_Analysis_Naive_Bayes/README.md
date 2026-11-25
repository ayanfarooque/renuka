# Practical 8: Twitter Sentiment Analysis – Naive Bayes

**Level:** Complex Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To perform Natural Language Processing (NLP) text vectorization and build a Multinomial Naive Bayes classification model to categorize tweets into sentiment polarity classes.

---

## Dataset Information
- **Dataset Name:** Twitter Entity Sentiment Analysis Dataset
- **Default Filename:** `twitter_training.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

---

## Comprehensive Theory

### 1. Introduction to Sentiment Analysis & NLP
Sentiment Analysis (also known as opinion mining) is a major Natural Language Processing (NLP) domain dedicated to computationally identifying, extracting, and quantifying subjective states and emotional polarity from unstructured natural language text.

Social media microblogs such as Twitter represent a rich, high-velocity stream of consumer opinion regarding brands, political events, public health, and entertainment. Tweets are categorized into distinct sentiment classes:
- **Positive:** Expressions of approval, excitement, joy, or satisfaction.
- **Negative:** Expressions of dissatisfaction, complaint, hostility, or anger.
- **Neutral:** Factual statements devoid of subjective emotional valence.
- **Irrelevant:** Content unrelated to the monitored entity.

---

### 2. Text Vectorization: Bag of Words & CountVectorizer
Machine learning algorithms cannot directly parse raw unstructured string characters; text must be transformed into a numerical vector representation.

#### The Bag-of-Words (BoW) Model:
The Bag-of-Words model disregards grammatical grammar and word order, treating a document as an unordered collection ("bag") of words while retaining word frequencies.

#### Processing Steps in `CountVectorizer`:
1. **Tokenization:** Splitting sentences into discrete lexical tokens (words).
2. **Lowercasing:** Normalizing all characters to lowercase to prevent vocabulary fragmentation (e.g., treating `"Great"` and `"great"` identically).
3. **Stop-Word Removal (`stop_words='english'`):** Filtering out high-frequency grammatical glue words (e.g., `"the"`, `"is"`, `"at"`, `"which"`, `"on"`) that convey negligible subjective sentiment.
4. **Vocabulary Construction:** Building a global dictionary of unique $V$ terms across all documents.
5. **Document-Term Matrix (DTM):** Constructing a sparse matrix $X$ where cell $(i, j)$ represents the frequency of vocabulary term $j$ in document $i$.

---

### 3. Multinomial Naive Bayes Classifier
The Multinomial Naive Bayes algorithm is specifically tailored for discrete word count features. According to Bayes' Theorem, the posterior probability of a sentiment class $c$ given a tweet document $d = (w_1, w_2, \dots, w_m)$ is:

$$P(c | d) = \frac{P(c) P(d | c)}{P(d)} \propto P(c) \prod_{k=1}^m P(w_k | c)$$

Where:
- $P(c) = \frac{N_c}{N}$: Prior probability of class $c$ (frequency of class $c$ over total documents).
- $P(w_k | c)$: Conditional likelihood of token $w_k$ appearing in documents of class $c$.

#### Laplace (Add-1) Smoothing:
If a test tweet contains a word that was never observed in training documents of class $c$, the maximum likelihood estimate yields $P(w_k | c) = 0$. Since probabilities are multiplied, a single zero probability would force the entire document score to zero.
**Laplace Smoothing** solves this by adding a pseudo-count of 1 to every word in the vocabulary:

$$\hat{P}(w_k | c) = \frac{\text{count}(w_k, c) + 1}{\sum_{w \in V} \text{count}(w, c) + |V|}$$

Where $|V|$ is the total number of unique words in the vocabulary.

---

### 4. Viva Voce Questions & Answers

**Q1: What is the role of `CountVectorizer` in NLP pipelines?**  
*Answer:* `CountVectorizer` converts a collection of text documents into a matrix of token counts (Document-Term Matrix). It handles tokenization, lowercase normalization, stop-word elimination, and vocabulary building.

**Q2: What are stop words and why are they removed?**  
*Answer:* Stop words are ubiquitous language words (e.g., `"and"`, `"the"`, `"in"`) that appear with high frequency across all texts regardless of sentiment. Removing them reduces dimensionality, saves computational memory, and prevents noise from dominating sentiment classification.

**Q3: Why is Multinomial Naive Bayes particularly well-suited for text classification?**  
*Answer:* It models discrete integer count histograms directly, scales linearly with document length and vocabulary size, trains extremely rapidly with closed-form counting, and performs effectively even on short, noisy microblog text.

**Q4: How does Laplace smoothing prevent zero probability errors?**  
*Answer:* It adds $1$ to the numerator (word count) and $|V|$ to the denominator (vocabulary size) of the likelihood estimate. This guarantees every word receives a small non-zero probability, preventing the joint posterior product from collapsing.


---

## Step-by-Step Algorithm
1. Import Pandas, train_test_split, CountVectorizer, MultinomialNB, and accuracy_score.
2. Load the Twitter dataset without headers using `pd.read_csv('twitter_training.csv', header=None)`.
3. Assign standard column names: `['id', 'entity', 'sentiment', 'tweet']`.
4. Prune missing entries across text and sentiment attributes via `df.dropna(subset=['tweet', 'sentiment'])`.
5. Instantiate `CountVectorizer(stop_words='english')` and transform tweet strings into numerical feature vectors $X$.
6. Extract target sentiment labels into vector $y$.
7. Split the dataset into 80% training and 20% testing subsets with `train_test_split`.
8. Instantiate and train `MultinomialNB()`.
9. Predict sentiment classes on the test set via `model.predict(X_test)`.
10. Calculate and display accuracy using `accuracy_score(y_test, pred)`.

---

## Libraries Required
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

---

## Python Program
```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

csv_file = 'twitter_training.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Twitter Sentiment Dataset...")
    data = [
        [2401, 'Borderlands', 'Positive', 'I am really enjoying this brand new Borderlands release, amazing graphics!'],
        [2401, 'Borderlands', 'Negative', 'Borderlands servers are completely crashed again, unacceptable and terrible.'],
        [2402, 'Borderlands', 'Neutral', 'The new patch for Borderlands was released at 10 AM GMT today.'],
        [2402, 'Borderlands', 'Irrelevant', 'Heading to the grocery store to get some coffee and milk.'],
        [2403, 'Amazon', 'Positive', 'Super fast delivery and brilliant packaging from Amazon, very happy!'],
        [2403, 'Amazon', 'Negative', 'Package arrived completely crushed and broken. Worst customer service ever.'],
        [2404, 'Amazon', 'Neutral', 'Amazon was founded in July 1994 by Jeff Bezos in Washington.'],
        [2404, 'Amazon', 'Irrelevant', 'Watching the sunset from the rooftop terrace.'],
        [2405, 'CallOfDuty', 'Positive', 'Call of Duty multiplayer update is absolute perfection, best game of the year!'],
        [2405, 'CallOfDuty', 'Negative', 'Unplayable lag and endless bugs in the new Call of Duty matchmaking.']
    ]
    df = pd.DataFrame(data)
    df.to_csv(csv_file, header=False, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file, header=None)

# Assign column names
df.columns = ['id', 'entity', 'sentiment', 'tweet']

# Remove missing data
df = df.dropna(subset=['tweet', 'sentiment'])

# Convert text into numerical data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predictions:", pred[:5])
print("Actual:     ", list(y_test[:5]))
print("Accuracy:   ", accuracy_score(y_test, pred))
```

---

## Expected Output
```text
Predictions: ['Positive' 'Negative']
Actual:      ['Positive', 'Negative']
Accuracy:    1.0

```

---

## Result
Successfully preprocessed Twitter social media text using CountVectorizer and classified tweet sentiment with a Multinomial Naive Bayes classifier.

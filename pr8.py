from pathlib import Path
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download required NLTK data (skip if already present)
def ensure_nltk_resource(resource_name: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(download_name, quiet=True)

ensure_nltk_resource("corpora/stopwords", "stopwords")

data_path = Path(__file__).with_name("Reviews.csv")
if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. Put Reviews.csv in the same folder as this script."
    )

df = pd.read_csv(data_path, engine='python', on_bad_lines='skip')

print("Dataset Shape:", df.shape)
print(df.head())

df = df[["Text", "Score"]].dropna()
df = df.rename(columns={"Text": "review", "Score": "rating"})
df = df.sample(10000, random_state=42).reset_index(drop=True)

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = wordpunct_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)

# Apply cleaning
df["clean_review"] = df["review"].apply(clean_text)

vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=10,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_review"])
print("Feature Matrix Shape:", X.shape)

lda = LatentDirichletAllocation(
    n_components=8,
    random_state=42,
    learning_method="batch"
)

lda.fit(X)

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        print(" | ".join([
            feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]
        ]))

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, 10)

topic_results = lda.transform(X)
df["topic"] = topic_results.argmax(axis=1)

print(df[["review", "topic"]].head())

df["topic"].value_counts().sort_index().plot(kind="bar", figsize=(10,5))
plt.title("Topic Distribution")
plt.xlabel("Topic")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("topic_distribution.png")
print("Saved: topic_distribution.png")

topic_sentiment = df.groupby("topic")["rating"].mean().sort_values(ascending=False)
print(topic_sentiment)

df.to_csv("customer_review_topics.csv", index=False)
print("Saved: customer_review_topics.csv")
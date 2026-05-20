# Import required libraries
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-learn is not installed. Run: python -m pip install scikit-learn"
    ) from exc

# Corpus
corpus = [
    "College life is a mix of stress and fun",
    "I often rush to class after waking up late",
    "Canteen food and friends make the day better",
    "Exams always come faster than expected",
    "Group projects usually mean one person does most of the work",
    "I spend more time on assignments than I planned",
    "Late night chats are better than early morning lectures",
    "I am trying to figure out my career path",
    "College is teaching me more than just academics",
    "These days will probably matter later in life"
]

print("Original Corpus:\n", corpus)

# 1. BAG OF WORDS (BoW)
print("\n--- Bag of Words (BoW) ---")
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus)

# Vocabulary
print("\nVocabulary:\n", bow_vectorizer.get_feature_names_out())

# Matrix form
print("\nBoW Matrix:\n", bow_matrix.toarray())

# 2. TF-IDF
print("Original Corpus:\n", corpus)
print("\n--- TF-IDF ---")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Vocabulary
print("\nVocabulary:\n", tfidf_vectorizer.get_feature_names_out())

# Matrix form
print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())
# Ayan Ahmad Farooque BT23F05F019

# Import libraries
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources (skip if already present)
def ensure_nltk_resource(resource_name: str, download_name: str) -> None:
	try:
		nltk.data.find(resource_name)
	except LookupError:
		nltk.download(download_name, quiet=True)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
ensure_nltk_resource('corpora/wordnet', 'wordnet')
ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4')

# Load spaCy model
try:
	nlp = spacy.load("en_core_web_sm")
except OSError as exc:
	raise OSError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from exc

# Sample text
text = """
I woke up early today for no reason.
The room was quiet, but something felt off.
My phone showed 5:00 AM, but outside it looked like midnight.
"""
print("Original Text:\n", text)

# 1. TOKENIZATION
print("\n--- Tokenization ---")
# Sentence Tokenization
sentences = sent_tokenize(text)
print("\nSentences:\n", sentences)
# Word Tokenization
words = word_tokenize(text)
print("\nWords:\n", words)

# 2. STEMMING (NLTK)
print("\n--- Stemming (Porter Stemmer) ---")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("\nStemmed Words:\n", stemmed_words)

# 3. LEMMATIZATION (NLTK)
print("\n--- Lemmatization (NLTK) ---")
lemmatizer = WordNetLemmatizer()
lemmatized_words_nltk = [lemmatizer.lemmatize(word) for word in words]
print("\nNLTK Lemmatized Words:\n", lemmatized_words_nltk)

# 4. LEMMATIZATION (spaCy)
print("\n--- Lemmatization (spaCy) ---")
doc = nlp(text)
lemmatized_words_spacy = [token.lemma_ for token in doc]
print("\nspaCy Lemmatized Words:\n", lemmatized_words_spacy)
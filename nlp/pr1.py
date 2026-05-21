# Import required libraries
import nltk
import re
import string
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

# Download required datasets (skip if already present)
def ensure_nltk_resource(resource_name: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(download_name, quiet=True)

ensure_nltk_resource('corpora/stopwords', 'stopwords')

# Sample text
text = """
Hi there! I'm Ayan Farooque. Feel free to reach out to me at ayfar@gmail.com
for any queries or collaborations. You can also check out my website at https://geca.com.
If needed, give me a call at 999999999.
I won't be able to join today's class as I am occupied with some work.
"""
print("Original Text:\n", text)

# 1. Lowercasing
text = text.lower()
print("\nAfter Lowercasing:\n", text)

# 2. Handling Contractions
contractions = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "you're": "you are",
    "don't": "do not"
}
for key, value in contractions.items():
    text = text.replace(key, value)
print("\nAfter Expanding Contractions:\n", text)

# 3. Extract data using Regex
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}', text)
phones = re.findall(r'\b\d{10}\b', text)
urls = re.findall(r'https?://\S+|www\.\S+', text)
print("\nExtracted Emails:", emails)
print("Extracted Phone Numbers:", phones)
print("Extracted URLs:", urls)

# 4. Remove punctuation
text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
print("\nAfter Removing Punctuation:\n", text_no_punct)

# 5. Tokenization
words = wordpunct_tokenize(text_no_punct)

# 6. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("\nFinal Processed Words:\n", filtered_words)
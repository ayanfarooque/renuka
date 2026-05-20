# Import libraries
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict

# Download tokenizer resources (skip if already present)
def ensure_nltk_resource(resource_name: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(download_name, quiet=True)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')

# Sample text
text = """
The education sector is experiencing a significant transformation as digital technologies continue to reshape traditional learning methods.
Schools, colleges, and universities are increasingly adopting online platforms, enabling students to access educational content from anywhere in the world.
The rise of e-learning has introduced flexibility in education, allowing learners to study at their own pace.
Hybrid models, combining both online and offline teaching, are becoming more common as institutions aim to provide a balanced and effective learning experience.
This shift has also encouraged the use of interactive tools such as virtual classrooms, recorded lectures, and AI-based learning systems.
In addition, educators are focusing more on skill-based learning to prepare students for real-world challenges.
Courses related to data science, artificial intelligence, and digital marketing are gaining popularity, reflecting the changing demands of the job market. Many institutions are also collaborating with industries to offer practical exposure through internships and projects.
"""
# Preprocess
text = text.lower()
tokens = word_tokenize(text)

# Generate Unigrams, Bigrams, Trigrams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))
print("UNIGRAMS:\n", unigrams[:20])
print("\nBIGRAMS:\n", bigrams[:20])
print("\nTRIGRAMS:\n", trigrams[:20])

# Function to build N-gram model
def build_ngram_model(tokens, n):
    model = defaultdict(list)

    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model[key].append(next_word)

    return model

# Function to generate text
def generate_text(model, n, start_words, num_words):
    current = tuple(start_words)
    output = list(start_words)
    for _ in range(num_words):
        if current in model:
            next_word = random.choice(model[current])
            output.append(next_word)
            current = tuple(output[-(n-1):])
        else:
            break

    return " ".join(output)

# Trigram Model
n = 3
model = build_ngram_model(tokens, n)
start = ["digital", "payments"]
generated = generate_text(model, n, start, 15)
print("\nGENERATED TEXT:\n", generated)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Source Knowledge Base
docs = [
    "Paris is the capital of France. The Eiffel Tower is in Paris. It was completed in 1889.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The Transformer architecture was introduced in 2017 in the paper Attention Is All You Need.",
    "GANs were introduced by Ian Goodfellow in 2014. They have a generator and a discriminator.",
    "RAG combines a retriever with a language model so answers are grounded in documents."
]

# 2. Document Chunking
chunks = [s.strip() for d in docs for s in re.split(r'(?<=\.)\s+', d) if s.strip()]
print(f"Total Chunks Created: {len(chunks)}")

# 3. Embeddings & Vector Store (TF-IDF Index)
vec = TfidfVectorizer(stop_words="english").fit(chunks)
index = vec.transform(chunks)

# 4. Semantic Retriever
def retrieve(q, k=2):
    q_vec = vec.transform([q])
    scores = cosine_similarity(q_vec, index)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [(chunks[i], scores[i]) for i in top_indices]

# 5. Prompt Engineering Template
def make_prompt(q, ctx):
    context_str = "\n- ".join(c for c, _ in ctx)
    return (
        "System: You are a careful assistant. Answer ONLY from the context provided. "
        "If the answer is not in the context, say 'I don't know'.\n"
        f"Context:\n- {context_str}\n"
        f"Question: {q}\n"
        "Answer:"
    )

# 6. Grounded Generator (prevents hallucinations when similarity < 0.15)
def llm_grounded_answer(q, ctx):
    best_chunk, best_score = ctx[0]
    if best_score > 0.15:
        return best_chunk
    return "I don't know"

# 7. Test Queries
queries = [
    "Who created Python?",
    "When was the Transformer introduced?",
    "Who is the president of Mars?"  # Unanswerable (tests hallucination refusal)
]

for q in queries:
    ctx = retrieve(q)
    print("--------------------------------------------------")
    print("Question:", q)
    print("Retrieved Passages:", [c for c, s in ctx if s > 0.15])
    print("Grounded Answer:   ", llm_grounded_answer(q, ctx))

print("--------------------------------------------------")
print("Engineered Prompt Example:\n")
print(make_prompt("Who invented GANs?", retrieve("Who invented GANs?")))\n
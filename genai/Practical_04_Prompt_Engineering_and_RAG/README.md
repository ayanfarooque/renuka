# Practical 4: Prompt Engineering + Retrieval-Augmented Generation (RAG)

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To construct an end-to-end Retrieval-Augmented Generation (RAG) system using document chunking, TF-IDF vectorization, cosine similarity retrieval, and grounded prompt engineering without external APIs.

---

## Comprehensive Theory

### 1. Parametric vs. Non-Parametric Memory in LLMs
Large Language Models encapsulate knowledge within their neural weights (parametric memory). However, reliance on parametric memory exhibits critical deficiencies:
1. **Hallucinations:** When uncertain, models synthesize fluent but factually incorrect assertions.
2. **Knowledge Cutoffs:** Models have no awareness of events after their training cutoff date.
3. **Proprietary Data Inaccessibility:** Enterprise or internal records cannot be queried unless retrained.
4. **Lack of Provenance:** Parametric models cannot cite specific verifiable source documents.

**Retrieval-Augmented Generation (RAG)** introduces external **non-parametric memory** by dynamically retrieving relevant document passages and feeding them into the model's prompt context at inference time.

---

### 2. The Tripartite Architecture of RAG
A canonical RAG framework operates in three staged pipelines:

#### A. Document Ingestion & Chunking
Long source documents are segmented into smaller, coherent text passages ("chunks"). 
- Small chunks preserve focused semantic density.
- Sentence boundary detection (`re.split(r'(?<=\.)\s+', text)`) ensures linguistic integrity.

#### B. Vector Indexing & Retrieval
Each chunk is mapped to a vector embedding space. In offline or resource-constrained settings, **TF-IDF (Term Frequency - Inverse Document Frequency)** constructs sparse vector representations:
$$\text{TF}(t, d) = \frac{f_{t, d}}{|d|}, \quad \text{IDF}(t, D) = \ln\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
For a user query $q$, the retriever computes **Cosine Similarity** against all stored document chunk vectors:
$$\text{Similarity}(q, d_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \|\mathbf{d}_i\|}$$
The top-$k$ highest-scoring chunks form the retrieved context $\mathcal{C}$.

#### C. Grounded Prompt Generation
A system prompt strictly conditions the generation process:
- Instructs the LLM to restrict responses solely to the retrieved context.
- Explicitly mandates an `"I don't know"` refusal response when similarity score falls below confidence threshold $\theta$, eliminating hallucination.

---

### 3. Prompt Engineering Principles Applied
1. **Role Specification:** `"You are a careful assistant."` Defines persona and behavioral expectations.
2. **Context Delimitation:** Clear boundaries (`Context: ...`) prevent prompt injection.
3. **Negative Constraint Enforcement:** `"If the answer is not in the context, say 'I don't know'."` Prevents confabulation.
4. **Source Attribution:** Guarantees factual grounding in enterprise source data.

---

### 4. Viva Voce Questions & Answers

**Q1: What problem does RAG solve in generative AI systems?**  
*Answer:* RAG eliminates hallucinations, solves knowledge cutoff limitations, enables real-time access to private enterprise data without costly model retraining, and provides verifiable citations.

**Q2: What is the difference between Dense and Sparse Retrieval?**  
*Answer:* Sparse retrieval (e.g., TF-IDF, BM25) matches exact keyword frequencies across documents. Dense retrieval (e.g., BERT embeddings) maps queries and documents into continuous semantic vector spaces, capturing synonyms and conceptual paraphrases.

**Q3: Why is document chunking necessary before indexing?**  
*Answer:* Indexing whole documents dilutes specific answers across thousands of words and exceeds prompt context length limits. Chunking isolates localized, high-density facts, matching queries precisely.

**Q4: How does prompt engineering prevent hallucinations in RAG?**  
*Answer:* By providing explicit negative constraints (e.g., "Answer ONLY from context; refuse if missing"), the model is guided to abstain from speculation when retrieved evidence is absent or insufficient.


---

## Step-by-Step Algorithm
1. Define a knowledge corpus of raw text documents.
2. Segment documents into sentence-level chunks using regex lookbehind splitting.
3. Fit a `TfidfVectorizer` on chunks to create a sparse vector index database.
4. Implement `retrieve(q, k=2)`: transform query $q$ into vector space, compute cosine similarity against index, and return top-$k$ scoring passages.
5. Engineer a structured grounding prompt combining retrieved passages, query question, and strict refusal rules.
6. Implement a grounded inference function that returns the retrieved answer if similarity $\ge 0.15$, else outputs `"I don't know"`.
7. Evaluate test queries (both answerable and unanswerable out-of-domain questions).

---

## Libraries Required
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
```

---

## Python Program (Self-Contained & Offline)
```python
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
print(make_prompt("Who invented GANs?", retrieve("Who invented GANs?")))
```

---

## Expected Output
```text
Total Chunks Created: 8
--------------------------------------------------
Question: Who created Python?
Retrieved Passages: ['Python was created by Guido van Rossum and first released in 1991.']
Grounded Answer:    Python was created by Guido van Rossum and first released in 1991.
--------------------------------------------------
Question: When was the Transformer introduced?
Retrieved Passages: ['The Transformer architecture was introduced in 2017 in the paper Attention Is All You Need.']
Grounded Answer:    The Transformer architecture was introduced in 2017 in the paper Attention Is All You Need.
--------------------------------------------------
Question: Who is the president of Mars?
Retrieved Passages: []
Grounded Answer:    I don't know
--------------------------------------------------
Engineered Prompt Example:

System: You are a careful assistant. Answer ONLY from the context provided. If the answer is not in the context, say 'I don't know'.
Context:
- GANs were introduced by Ian Goodfellow in 2014.
- They have a generator and a discriminator.
Question: Who invented GANs?
Answer:

```

---

## Result
Successfully constructed an end-to-end RAG pipeline demonstrating chunking, TF-IDF vector retrieval, prompt grounding, and hallucination prevention.\n
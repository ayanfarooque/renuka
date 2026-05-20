#pip install gensim
try:
    from gensim.models import Word2Vec
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "gensim is not installed. Run: python -m pip install gensim"
    ) from exc
# More sentences on Banking Topic
sentences = [
    ["banking", "is", "advanced", "nowadays"],
    ["customers", "use", "mobile", "banking", "apps"],
    ["digital", "payments", "are", "fast", "and", "secure"],
    ["banks", "provide", "online", "money", "transfer"],
    ["artificial", "intelligence", "helps", "fraud", "detection"],
    ["internet", "banking", "is", "easy", "to", "use"],
    ["cashless", "transactions", "are", "common", "today"],
    ["modern", "banks", "offer", "smart", "services"],
    ["users", "check", "account", "balance", "online"],
    ["secure", "systems", "improve", "banking", "safety"]
]
# Train Word2Vec Model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
# Print vectors for multiple words
print("Vector for 'banking':\n", model.wv["banking"])
print("\nVector for 'digital':\n", model.wv["digital"])
print("\nVector for 'banks':\n", model.wv["banks"])
# Print similar words for multiple inputs
print("\nMost similar words to 'banking':")
print(model.wv.most_similar("banking"))
print("\nMost similar words to 'digital':")
print(model.wv.most_similar("digital"))
print("\nMost similar words to 'banks':")
print(model.wv.most_similar("banks"))
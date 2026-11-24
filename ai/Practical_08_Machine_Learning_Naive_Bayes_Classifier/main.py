data = [
    (["Sunny", "Hot"], "No"),
    (["Sunny", "Cool"], "No"),
    (["Cloudy", "Hot"], "Yes"),
    (["Cloudy", "Cool"], "Yes"),
    (["Cloudy", "Hot"], "Yes"),
    (["Sunny", "Hot"], "No")
]
query = ["Sunny", "Cool"]
classes = ["Yes", "No"]

def predict(query):
    scores = {}
    for c in classes:
        rows = [x for x, y in data if y == c]
        score = len(rows) / len(data)
        for i, value in enumerate(query):
            count = sum(row[i] == value for row in rows)
            score *= (count + 1) / (len(rows) + 2)
        scores[c] = score
    return max(scores, key=scores.get)

print("Prediction:", predict(query))

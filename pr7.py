from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
# Make sure file is in this folder OR give full path
data_path = Path(__file__).with_name("training.1600000.processed.noemoticon.csv")
if not data_path.exists():
  raise FileNotFoundError(
    f"Dataset not found at {data_path}. Put the CSV in the same folder as this script."
  )

data = pd.read_csv(
  data_path,
    encoding='latin-1',
    header=None,
    names=['label', 'id', 'date', 'query', 'user', 'text'],
    engine='python', # Added for robust parsing
    on_bad_lines='skip' # Skip malformed lines
)
print(data.head())

# Step 2: Convert labels
# 0 = negative, 4 = positive
data['label'] = data['label'].map({0: 0, 4: 1})

# Step 3: Reduce size (faster training)
data = data.sample(10000, random_state=42)

# Step 4: Features & target
X = data['text']
y = data['label']

# Step 5: Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# Predictions
nb_pred = nb_model.predict(X_test)

# Accuracy
print("Naïve Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("\n")

# Step 8: Prediction function
def predict_sentiment(model, vectorizer, text):
  vect = vectorizer.transform([text])
  pred = model.predict(vect)
  return "positive" if pred[0] == 1 else "negative"

# Step 9: Test prediction
sample = "I love this product! It's amazing."
print(sample)
print("Prediction:", predict_sentiment(nb_model, vectorizer, sample))
sample = "Worst product ever!"
print(sample)
print("Prediction:", predict_sentiment(nb_model, vectorizer, sample))
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

csv_file = 'twitter_training.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Twitter Sentiment Dataset...")
    data = [
        [2401, 'Borderlands', 'Positive', 'I am really enjoying this brand new Borderlands release, amazing graphics!'],
        [2401, 'Borderlands', 'Negative', 'Borderlands servers are completely crashed again, unacceptable and terrible.'],
        [2402, 'Borderlands', 'Neutral', 'The new patch for Borderlands was released at 10 AM GMT today.'],
        [2402, 'Borderlands', 'Irrelevant', 'Heading to the grocery store to get some coffee and milk.'],
        [2403, 'Amazon', 'Positive', 'Super fast delivery and brilliant packaging from Amazon, very happy!'],
        [2403, 'Amazon', 'Negative', 'Package arrived completely crushed and broken. Worst customer service ever.'],
        [2404, 'Amazon', 'Neutral', 'Amazon was founded in July 1994 by Jeff Bezos in Washington.'],
        [2404, 'Amazon', 'Irrelevant', 'Watching the sunset from the rooftop terrace.'],
        [2405, 'CallOfDuty', 'Positive', 'Call of Duty multiplayer update is absolute perfection, best game of the year!'],
        [2405, 'CallOfDuty', 'Negative', 'Unplayable lag and endless bugs in the new Call of Duty matchmaking.']
    ]
    df = pd.DataFrame(data)
    df.to_csv(csv_file, header=False, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file, header=None)

# Assign column names
df.columns = ['id', 'entity', 'sentiment', 'tweet']

# Remove missing data
df = df.dropna(subset=['tweet', 'sentiment'])

# Convert text into numerical data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predictions:", pred[:5])
print("Actual:     ", list(y_test[:5]))
print("Accuracy:   ", accuracy_score(y_test, pred))

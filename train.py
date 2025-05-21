import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from utils import preprocess_text

# Load data
data = pd.read_csv("data/spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Preprocess
data['text'] = data['text'].apply(preprocess_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc:.2f}")

# Save model and vectorizer
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

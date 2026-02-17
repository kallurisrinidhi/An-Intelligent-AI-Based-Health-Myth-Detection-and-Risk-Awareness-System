import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

data = pd.read_csv("health_myths.csv")

print("Dataset loaded successfully")
print("Total records:", len(data))

X = data["belief_text"]
y = data["label"]  # 1 = FACT, 0 = MYTH

# Split dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Converting text to numbers using TF-IDF with bigrams...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # 🔥 BIG ACCURACY BOOST
    max_df=0.95,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Machine Learning model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"  # 🔥 Handles class imbalance
)

model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Saving model and vectorizer...")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")

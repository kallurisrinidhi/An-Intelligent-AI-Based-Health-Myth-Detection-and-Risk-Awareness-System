import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Loading dataset...")

# ================= LOAD DATA =================
data = pd.read_csv("health_myths.csv")

# ================= LOAD MODEL =================
print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

true_labels = []
predicted_labels = []

print("Running Leave-One-Out evaluation...")

# ================= LEAVE ONE OUT TESTING =================
for i in range(len(data)):

    # Remove current row to avoid self-matching
    temp_data = data.drop(i).reset_index(drop=True)

    # Encode remaining dataset
    temp_embeddings = model.encode(
        temp_data["belief_text"].tolist(),
        convert_to_tensor=True
    )

    belief = data.iloc[i]["belief_text"]
    true_label = data.iloc[i]["label"]

    # Encode user input
    user_embedding = model.encode(belief, convert_to_tensor=True)

    # Compute similarity
    similarities = util.cos_sim(user_embedding, temp_embeddings)

    best_match_index = similarities.argmax().item()

    predicted_label = temp_data.iloc[best_match_index]["label"]

    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# ================= METRICS =================
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

print("\n========= MODEL PERFORMANCE =========")
print("Accuracy :", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall   :", round(recall,3))
print("F1 Score :", round(f1,3))

print("\nConfusion Matrix:")
print(cm)
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test inputs
test_beliefs = [
    "Vaccines cause infertility",
    "Exercise improves heart health",
    "Drinking hot water cures COVID",
    "Smoking causes lung cancer"
]

print("\nTesting trained model:\n")

for belief in test_beliefs:
    vector = vectorizer.transform([belief])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        print(f"{belief}  --> FACT")
    else:
        print(f"{belief}  --> MYTH")

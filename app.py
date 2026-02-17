from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset
data = pd.read_csv("health_myths.csv")

CONFIDENCE_THRESHOLD = 60

# Prepare vectors for similarity comparison (PHASE C)
dataset_texts = data["belief_text"].tolist()
dataset_vectors = vectorizer.transform(dataset_texts)

# Risk base scores (PHASE B)
RISK_BASE = {
    "Low": 30,
    "Medium": 60,
    "High": 85
}

@app.route("/", methods=["GET", "POST"])
def index():
    # -------- DEFAULT VALUES (VERY IMPORTANT) --------
    result = None
    explanation = None
    risk = None
    confidence = None
    color = None
    advisory = None
    risk_score = None
    impact = None
    closest_belief = "N/A"
    similarity_score = 0

    if request.method == "POST":
        belief = request.form["belief"]

        vector = vectorizer.transform([belief])
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0].max()
        confidence = round(prob * 100, 2)

        # ---------- PHASE A: UNCERTAINTY HANDLING ----------
        if confidence < CONFIDENCE_THRESHOLD:
            result = "UNCERTAIN"
            color = "uncertain"
            explanation = "AI confidence is low due to limited similarity with known data."
            risk = "Medium"
            advisory = "Consult a medical professional."
            impact = "Potential misinformation impact"
            risk_score = 60

        else:
            if prediction == 1:
                result = "FACT"
                color = "fact"
            else:
                result = "MYTH"
                color = "myth"

            # ---------- PHASE C: NLP SIMILARITY ENGINE ----------
            user_vector = vectorizer.transform([belief])
            similarities = cosine_similarity(user_vector, dataset_vectors)

            best_match_index = similarities.argmax()
            similarity_score = round(similarities[0][best_match_index] * 100, 2)

            closest_row = data.iloc[best_match_index]
            closest_belief = closest_row["belief_text"]
            explanation = closest_row["explanation"]
            risk = closest_row["risk"]

            # ---------- PHASE B: RISK SEVERITY SCORING ----------
            base_score = RISK_BASE.get(risk, 60)
            confidence_boost = confidence * 0.2
            risk_score = int(base_score + confidence_boost)

            if result == "MYTH":
                risk_score += 10
                impact = "Delay in proper medical treatment"
            else:
                impact = "Minimal health risk"

            risk_score = min(risk_score, 100)

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        risk=risk,
        confidence=confidence,
        color=color,
        advisory=advisory,
        risk_score=risk_score,
        impact=impact,
        closest_belief=closest_belief,
        similarity_score=similarity_score
    )
@app.route("/feedback", methods=["POST"])
def feedback():
    belief = request.form["belief"]
    ai_result = request.form["ai_result"]
    confidence = request.form["confidence"]
    user_feedback = request.form["user_feedback"]

    feedback_data = {
        "belief": belief,
        "ai_result": ai_result,
        "confidence": confidence,
        "feedback": user_feedback
    }

    df = pd.DataFrame([feedback_data])
    df.to_csv("feedback.csv", mode="a", header=False, index=False)

    return "Feedback recorded successfully"
@app.route("/analytics")
def analytics():
    try:
        feedback_df = pd.read_csv("feedback.csv")
    except:
        feedback_df = pd.DataFrame(columns=["belief", "ai_result", "confidence", "feedback"])

    total_queries = len(feedback_df)

    myth_count = len(feedback_df[feedback_df["ai_result"] == "MYTH"])
    fact_count = len(feedback_df[feedback_df["ai_result"] == "FACT"])

    correct_count = len(feedback_df[feedback_df["feedback"] == "Correct"])
    incorrect_count = len(feedback_df[feedback_df["feedback"] == "Incorrect"])

    avg_confidence = round(feedback_df["confidence"].astype(float).mean(), 2) if total_queries > 0 else 0

    return render_template(
        "analytics.html",
        total_queries=total_queries,
        myth_count=myth_count,
        fact_count=fact_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        avg_confidence=avg_confidence
    )


if __name__ == "__main__":
    app.run(debug=True)

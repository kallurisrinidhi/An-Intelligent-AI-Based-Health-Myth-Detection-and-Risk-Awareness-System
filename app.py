from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# ================= LOAD DATA =================

data = pd.read_csv("health_myths.csv")

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode dataset once (VERY IMPORTANT)
dataset_embeddings = model.encode(
    data["belief_text"].tolist(),
    convert_to_tensor=True
)

CONFIDENCE_THRESHOLD = 60

RISK_BASE = {
    "Low": 30,
    "Medium": 60,
    "High": 85
}

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= LIBRARY PAGE =================
@app.route("/library")
def library():
    return render_template("library.html")

# ================= ABOUT PAGE =================
@app.route("/about")
def about():
    return render_template("about.html")

# ================= ANALYZE =================
@app.route("/analyze", methods=["POST"])
def analyze():

    belief = request.form["belief"]

    # Encode user sentence
    user_embedding = model.encode(belief, convert_to_tensor=True)

    # Semantic similarity
    similarities = util.cos_sim(user_embedding, dataset_embeddings)

    best_match_index = similarities.argmax().item()
    similarity_score = float(similarities[0][best_match_index]) * 100

    closest_row = data.iloc[best_match_index]

    explanation = closest_row["explanation"]
    risk = closest_row["risk"]
    label = closest_row["label"]

    result = "FACT" if label == 1 else "MYTH"
    color = "fact" if label == 1 else "myth"

    confidence = round(similarity_score,2)

    # Risk score logic
    if confidence < CONFIDENCE_THRESHOLD:
        result = "UNCERTAIN"
        color = "uncertain"
        explanation = "Low AI confidence due to unfamiliar pattern."
        risk_score = 60
    else:
        base_score = RISK_BASE.get(risk,60)
        risk_score = min(int(base_score + confidence*0.2),100)

    # ⭐ Save history
    history = pd.DataFrame([[belief,result,confidence]],
                           columns=["belief","result","confidence"])
    history.to_csv("history.csv",mode="a",header=False,index=False)

    return jsonify({
        "result":result,
        "confidence":confidence,
        "explanation":explanation,
        "risk_score":risk_score,
        "color":color
    })

# ================= FEEDBACK SAVE =================
@app.route("/feedback", methods=["POST"])
def feedback():

    belief = request.form["belief"]
    ai_result = request.form["ai_result"]
    confidence = request.form["confidence"]
    user_feedback = request.form["feedback"]

    row = pd.DataFrame([[belief,ai_result,confidence,user_feedback]],
                       columns=["belief","ai_result","confidence","feedback"])

    row.to_csv("feedback.csv",mode="a",header=False,index=False)

    return jsonify({"status":"saved"})

# ================= ANALYTICS =================
@app.route("/analytics")
def analytics():

    try:
        feedback_df = pd.read_csv("feedback.csv")
    except:
        feedback_df = pd.DataFrame(columns=["belief","ai_result","confidence","feedback"])

    total_queries = len(feedback_df)
    myth_count = len(feedback_df[feedback_df["ai_result"]=="MYTH"])
    fact_count = len(feedback_df[feedback_df["ai_result"]=="FACT"])
    correct_count = len(feedback_df[feedback_df["feedback"]=="Correct"])
    incorrect_count = len(feedback_df[feedback_df["feedback"]=="Incorrect"])

    avg_confidence = round(
        feedback_df["confidence"].astype(float).mean(),2
    ) if total_queries>0 else 0

    return render_template(
        "analytics.html",
        total_queries=total_queries,
        myth_count=myth_count,
        fact_count=fact_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        avg_confidence=avg_confidence
    )

# ================= ADMIN =================
@app.route("/admin")
def admin():

    try:
        history = pd.read_csv("history.csv")
    except:
        history = pd.DataFrame(columns=["belief","result","confidence"])

    total = len(history)

    return render_template("admin.html", total_queries=total)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
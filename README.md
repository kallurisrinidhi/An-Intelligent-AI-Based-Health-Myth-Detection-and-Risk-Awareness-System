#AI-Based Health Myth Detection and Risk Awareness System
Project Description

This project is a full-stack Flask web application that classifies health-related claims as FACT or MYTH and generates a risk level for misleading statements. Instead of relying on a single model, it combines semantic similarity, a trained classifier, and rule-based logic into one hybrid prediction pipeline — along with a live analytics dashboard to track how the system performs over time.

The project covers the complete pipeline, including:

Data Loading & Preprocessing
Semantic Embedding Generation (Sentence-BERT)
Classifier Training & Evaluation
Hybrid Prediction (Embeddings + Classifier + Rule-Based Correction)
Confidence Scoring & Risk-Level Classification
Analytics Dashboard & Visualization
User Feedback Logging
Authentication (Admin Access)
Deployment Configuration
Project Structure
text
Health-Myth-Detection/
│
├── data/
│   └── dataset.csv
│
├── model/
│   └── logistic_model.joblib
│
├── templates/
│   ├── index.html
│   ├── login.html
│   └── analytics.html
│
├── static/
│   └── style.css
│
├── app.py
├── model_utils.py
├── train_model.py
├── download_model.py
├── check_dataset.py
├── verify_dataset.py
├── test_app.py
├── test_dataset.py
├── history.csv
├── requirements.txt
├── Procfile
└── README.md

Technologies Used
Python
Flask
Sentence-Transformers (all-mpnet-base-v2)
Scikit-learn (Logistic Regression, TF-IDF)
Pandas, NumPy
Joblib
Chart.js

How It Works — Hybrid Prediction Pipeline
Preprocessing — user input and dataset text are cleaned (lowercased, punctuation stripped, spelling-corrected where available).
Semantic Similarity — the claim is embedded using a Sentence-BERT model (all-mpnet-base-v2) and compared against a labeled dataset using cosine similarity to find the top-5 closest matches.
Classifier Score — a TF-IDF + Logistic Regression classifier trained on the same dataset provides a second, independent prediction.
Rule-Based Correction — keyword-based positive/negative indicators can override the model output for clear-cut cases (e.g. explicit "safe"/"harmful" language).
Confidence & Risk Scoring — the final prediction is paired with a confidence percentage and a LOW / MEDIUM / HIGH risk level for MYTH results.
Explanation — the closest matching dataset entry's explanation is surfaced alongside the prediction for transparency.
Dataset

A curated dataset of 144 labeled health claims, with the following columns:

Column	Description
belief_text	The health claim text
label	FACT or MYTH
explanation	Reasoning behind the label
risk_level	Associated risk level

Class distribution: 70 FACT / 74 MYTH

Model Training

train_model.py trains a Logistic Regression classifier on Sentence-BERT embeddings of the dataset and evaluates it using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

The trained classifier is saved as logistic_model.joblib and loaded at runtime for hybrid predictions.

Features
Claim Checker — submit any health-related statement and get an instant FACT/MYTH prediction with confidence and risk level.
Analytics Dashboard (/analytics) — visualizes total queries, FACT/MYTH distribution, average confidence, and recent query history using Chart.js (pie, bar, and line charts).
Feedback Loop — users can flag predictions as correct/incorrect, logged to feedback.csv for future retraining.
Admin Login — session-based authentication guarding the analytics view.
Query History Logging — every prediction is timestamped and stored in history.csv.
Routes
Route	Method	Description
/	GET	Home page — submit a claim
/predict	POST	Runs the hybrid prediction pipeline
/login	GET, POST	Admin login
/logout	GET	Clears the session
/admin	GET	Redirects to analytics (auth required)
/analytics	GET	Analytics dashboard
/feedback	POST	Logs user feedback on a prediction
How to Run

Clone the repository

bash
git clone https://github.com/kallurisrinidhi/Health-Myth-Detection.git

Go to the project folder

bash
cd Health-Myth-Detection

Install dependencies

bash
pip install -r requirements.txt

bash
python download_model.py

Train the classifier (generates logistic_model.joblib)

bash
python train_model.py

Run the app

bash
python app.py

The app will be available at http://localhost:5000.

Author

Kalluri Srinidhi Reddy B.Tech (Artificial Intelligence & Machine Learning)

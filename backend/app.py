"""
app.py
------
Flask API that serves sentiment predictions from the trained model.

Endpoints:
    GET  /api/health           -> health check
    POST /api/predict          -> { "text": "..." } -> prediction JSON

Run:
    python3 app.py
Then open frontend/index.html in a browser (it calls this API at
http://localhost:5000).
"""

import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)  # allow the frontend (served from a different origin/file) to call this API

# ---------------------------------------------------------------------
# Load trained model + vectorizer once at startup
# ---------------------------------------------------------------------
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty 'text'."}), 400

    # --- ML model prediction (trained on the Kaggle-style dataset) ---
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    ml_label = model.predict(vec)[0]
    ml_probs = model.predict_proba(vec)[0]
    ml_confidence = float(max(ml_probs))

    # --- TextBlob cross-check (fast lexicon-based sanity check) ---
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        tb_label = "positive"
    elif polarity < -0.05:
        tb_label = "negative"
    else:
        tb_label = "neutral"

    return jsonify({
        "text": text,
        "ml_model": {
            "sentiment": ml_label,
            "confidence": round(ml_confidence, 4),
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(model.classes_, ml_probs)
            }
        },
        "textblob": {
            "sentiment": tb_label,
            "polarity": round(polarity, 4)
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

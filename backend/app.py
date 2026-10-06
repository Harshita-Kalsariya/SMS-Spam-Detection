"""Flask backend: loads the trained model and serves the frontend + /predict API."""
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "spam_model.joblib"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

if not MODEL_PATH.exists():
    raise SystemExit("spam_model.joblib not found. Run notebook/spam_detector.ipynb first.")
model = joblib.load(MODEL_PATH)


def clean_text(text: str) -> str:
    """Same cleaning as in the notebook (must stay identical)."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d{5,}", " longnumber ", text)
    text = re.sub(r"[^a-z0-9£$\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Message is empty."), 400
    if len(message) > 2000:
        return jsonify(error="Message is too long (max 2000 characters)."), 400

    spam_prob = float(model.predict_proba([clean_text(message)])[0][1])
    return jsonify(
        label="spam" if spam_prob >= 0.5 else "ham",
        spam_probability=round(spam_prob, 4),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
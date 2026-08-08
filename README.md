# Verdict — Text Sentiment Analyzer

A full sentiment analysis project: a trained ML classifier served over a
Flask API, plus a small website (HTML/CSS/JS) that calls it and shows the
result live.

```
sentiment-analyzer/
├── backend/
│   ├── app.py               # Flask API (POST /api/predict)
│   ├── train_model.py       # Trains + evaluates the classifier
│   ├── requirements.txt
│   ├── data/
│   │   └── dataset.csv      # Labeled dataset (text, sentiment)
│   └── model/
│       ├── sentiment_model.pkl
│       └── vectorizer.pkl
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## About the dataset (and the Kaggle question)

This was built to run in a sandboxed environment without general internet
access —dataset `backend/data/dataset.csv` was
generated from **NLTK's built-in `movie_reviews` corpus**: 2,000 hand-labeled
positive/negative movie reviews. It's the same underlying source many
Kaggle "IMDB movie review sentiment" datasets are built from, so the project
structure, training pipeline, and evaluation are all genuine.

**To swap in a real Kaggle dataset yourself** (recommended if you want
better real-world accuracy, e.g.
[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) or the
[IMDB 50K Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)):

1. Download the CSV from Kaggle.
2. Make sure it has a text column (`text`/`review`/`sentence`) and a label
   column (`sentiment`/`label`, with values like `positive`/`negative` or
   `1`/`0`).
3. Save it as `backend/data/dataset.csv`, replacing the existing file.
4. Re-run `python3 train_model.py`. The loader in `train_model.py` auto-detects
   common column names, so no code changes are needed for most Kaggle
   sentiment CSVs.

## Setup

```bash
cd sentiment-analyzer/backend
pip install -r requirements.txt
```

The first run of `train_model.py` needs the NLTK corpus once:


## Train the model

```bash
python3 train_model.py
```

This cleans the text, vectorizes it with TF-IDF (unigrams + bigrams),
trains a Logistic Regression classifier, prints accuracy/precision/recall,
and saves `model/sentiment_model.pkl` + `model/vectorizer.pkl`.

Current held-out test performance: **~84% accuracy** on a balanced
pos/neg test split.

## Run the API

```bash
python3 app.py
```

Starts a Flask server at `http://localhost:5000`.

- `GET /api/health` → `{"status": "ok"}`
- `POST /api/predict` with `{"text": "..."}` → sentiment label, model
  confidence, and full class probabilities, plus a fast TextBlob
  cross-check.

## Run the website

Just open `frontend/index.html` in a browser while `app.py` is running.
Paste in text, hit **Analyze sentiment**, and the page calls the API and
renders the result — a verdict stamp, confidence, and a probability bar per
class.

If you serve the frontend from somewhere other than a local file (e.g. a
static file server), update `API_BASE` at the top of `frontend/script.js`
to point at wherever `app.py` is running.

## Known limitations

- **Binary training labels.** The movie review corpus only has
  positive/negative labels, so the ML model itself never predicts
  "neutral" — only the TextBlob cross-check can. A dataset with an explicit
  neutral class (several exist on Kaggle) would fix this.
- **Domain + length mismatch.** The model is trained on full-length movie
  reviews, so very short, non-movie-related sentences (e.g. product or
  service reviews) sometimes get low-confidence or wrong predictions. The
  confidence score is a useful signal here — low confidence (near 50%)
  usually means the model is genuinely unsure, not confidently wrong.
- Swapping in a larger, more general Kaggle dataset (Sentiment140's 1.6M
  tweets, or an Amazon/product-review dataset) and re-running
  `train_model.py` is the most direct way to improve real-world accuracy.

## Tech stack

- **Backend:** Python, Flask, scikit-learn (TF-IDF + Logistic Regression),
  NLTK (dataset source), TextBlob (secondary lexicon-based check)
- **Frontend:** vanilla HTML/CSS/JS, no build step

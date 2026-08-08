"""
train_model.py
--------------
Trains a 3-class sentiment classifier on data/dataset.csv.

Classes:
    positive
    negative
    neutral

The script saves:
    model/sentiment_model.pkl
    model/vectorizer.pkl

It uses a combination of word and character TF-IDF features so that
the classifier can generalize better to short/unseen text.
"""

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/dataset.csv")
MODEL_PATH = Path("model/sentiment_model.pkl")
VECTORIZER_PATH = Path("model/vectorizer.pkl")

REQUIRED_CLASSES = {"negative", "neutral", "positive"}


def clean_text(text: str) -> str:
    """
    Normalize text while preserving useful sentiment words.

    Important:
    - We DO NOT remove English stop words.
    - Negation words such as 'not', 'never', and 'no' are retained.
    - Apostrophes are converted to spaces so contractions remain readable.
    """
    text = str(text).lower()

    # Keep letters, apostrophes and spaces; remove numbers/punctuation.
    text = re.sub(r"[^a-z'\s]", " ", text)

    # Make common contractions explicit enough for TF-IDF.
    replacements = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "isn't": "is not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "never": "never",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)

    # Accept common column names.
    col_map = {}
    for col in df.columns:
        lower = str(col).lower().strip()

        if lower in ("text", "review", "sentence", "comment"):
            col_map[col] = "text"
        elif lower in ("sentiment", "label", "target", "polarity"):
            col_map[col] = "sentiment"

    df = df.rename(columns=col_map)

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "CSV must contain text/review/sentence/comment and "
            "sentiment/label/target/polarity columns."
        )

    df = df[["text", "sentiment"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

    # Normalize common label encodings.
    label_map = {
        "1": "positive",
        "pos": "positive",
        "positive": "positive",
        "0": "negative",
        "neg": "negative",
        "negative": "negative",
        "2": "neutral",
        "neu": "neutral",
        "neutral": "neutral",
    }

    df["sentiment"] = df["sentiment"].map(lambda x: label_map.get(x, x))

    # Remove unknown labels.
    before = len(df)
    df = df[df["sentiment"].isin(REQUIRED_CLASSES)].copy()
    removed = before - len(df)

    # Remove empty texts and duplicate text/label pairs.
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)

    if removed:
        print(f"Removed {removed} rows with unknown labels.")

    return df


def build_vectorizer():
    """
    Word + character TF-IDF.

    Word features learn sentiment-bearing words/phrases.
    Character features help with short/unseen wording and morphology.
    """
    word_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        lowercase=False,  # clean_text() already lowercases
        max_df=0.98,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        max_features=12000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )

    return FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ])


def main():
    print("=" * 60)
    print("3-CLASS SENTIMENT MODEL TRAINING")
    print("=" * 60)

    print("\nLoading dataset...")
    df = load_dataset(DATA_PATH)

    print(f"Loaded {len(df)} labeled examples.\n")
    print("Class distribution:")
    print(df["sentiment"].value_counts().sort_index())

    # Verify all three classes are present.
    missing = REQUIRED_CLASSES - set(df["sentiment"].unique())
    if missing:
        raise ValueError(f"Missing required class(es): {sorted(missing)}")

    # Warn if classes are substantially imbalanced.
    counts = df["sentiment"].value_counts()
    if counts.max() / counts.min() > 1.20:
        print("\nWARNING: class imbalance detected.")
        print("The classifier will use class_weight='balanced'.")

    print("\nCleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Stratified split keeps the three classes represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples:  {len(X_test)}")

    print("\nBuilding word + character TF-IDF features...")
    vectorizer = build_vectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training feature matrix: {X_train_vec.shape}")
    print(f"Testing feature matrix:  {X_test_vec.shape}")

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model.fit(X_train_vec, y_train)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=["negative", "neutral", "positive"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix")
    print("(rows = actual, columns = predicted)")
    print("Labels: [negative, neutral, positive]")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=["negative", "neutral", "positive"],
    ))

    print("\nSaving model and vectorizer...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved model:      {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")

    # ---------------------------------------------------------------
    # Sanity tests: these are NOT used for training.
    # They are only there to immediately expose obvious failures.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY TESTS")
    print("=" * 60)

    test_sentences = [
        (
            "This is one of the best products I have ever purchased.",
            "positive",
        ),
        (
            "The movie dragged on forever and the plot made no sense.",
            "negative",
        ),
        (
            "The movie was released in 1998 and has a running time of 120 minutes.",
            "neutral",
        ),
        (
            "The food was delicious and the atmosphere was perfect for a date night.",
            "positive",
        ),
        (
            "The service was terrible and the staff were rude.",
            "negative",
        ),
        (
            "The laptop has 16GB of RAM and a 512GB SSD.",
            "neutral",
        ),
    ]

    for sentence, expected in test_sentences:
        cleaned = clean_text(sentence)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        scores = dict(zip(model.classes_, probabilities))

        status = "PASS" if prediction == expected else "CHECK"

        print(f"\n[{status}]")
        print(f"Text:     {sentence}")
        print(f"Expected: {expected}")
        print(f"Predicted:{prediction}")
        print(
            "Scores:   "
            f"negative={scores.get('negative', 0):.3f}, "
            f"neutral={scores.get('neutral', 0):.3f}, "
            f"positive={scores.get('positive', 0):.3f}"
        )


if __name__ == "__main__":
    main()
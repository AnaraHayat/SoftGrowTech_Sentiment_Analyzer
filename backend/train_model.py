"""
train_model.py
--------------
Train the 3-class sentiment classifier using the balanced dataset.

Classes: positive, negative, neutral
Outputs:
    model/sentiment_model.pkl
    model/vectorizer.pkl
"""

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use the balanced dataset instead of the old heavily imbalanced dataset.
DATA_PATH = Path("data/dataset_balanced.csv")
MODEL_PATH = Path("model/sentiment_model.pkl")
VECTORIZER_PATH = Path("model/vectorizer.pkl")
REQUIRED_CLASSES = {"negative", "neutral", "positive"}


def clean_text(text: str) -> str:
    """Normalize text while preserving sentiment and negation words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)

    replacements = {
        "can't": "can not", "cannot": "can not", "won't": "will not",
        "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "isn't": "is not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentence": "text", "comment": "text",
                            "label": "sentiment", "target": "sentiment", "polarity": "sentiment"})

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain text and sentiment columns.")

    df = df[["text", "sentiment"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

    label_map = {
        "1": "positive", "pos": "positive", "positive": "positive",
        "0": "negative", "neg": "negative", "negative": "negative",
        "2": "neutral", "neu": "neutral", "neutral": "neutral",
    }
    df["sentiment"] = df["sentiment"].map(lambda x: label_map.get(x, x))
    df = df[df["sentiment"].isin(REQUIRED_CLASSES)]
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)

    return df


def build_vectorizer():
    word_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        lowercase=False,
        max_df=0.98,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        max_features=12000,
        ngram_range=(3, 5),
        min_df=1,
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

    df = load_dataset(DATA_PATH)
    print(f"\nLoaded {len(df)} labeled examples.")
    print("\nClass distribution:")
    print(df["sentiment"].value_counts().sort_index())

    missing = REQUIRED_CLASSES - set(df["sentiment"].unique())
    if missing:
        raise ValueError(f"Missing required class(es): {sorted(missing)}")

    counts = df["sentiment"].value_counts()
    if counts.nunique() != 1:
        raise ValueError("Training dataset must be balanced: each class should have the same number of examples.")

    print("\nCleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"],
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

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight=None,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(
        y_test, y_pred,
        labels=["negative", "neutral", "positive"],
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix")
    print(confusion_matrix(
        y_test, y_pred,
        labels=["negative", "neutral", "positive"],
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nSaved model:      {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")

    print("\n" + "=" * 60)
    print("SANITY TESTS")
    print("=" * 60)

    test_sentences = [
        ("This is one of the best products I have ever purchased.", "positive"),
        ("The movie dragged on forever and the plot made no sense.", "negative"),
        ("The movie was released in 1998 and has a running time of 120 minutes.", "neutral"),
        ("The food was delicious and the atmosphere was perfect for a date night.", "positive"),
        ("The service was terrible and the staff were rude.", "negative"),
        ("The laptop has 16GB of RAM and a 512GB SSD.", "neutral"),
        ("I absolutely love this product and would recommend it to everyone.", "positive"),
        ("I absolutely hate this product and would never recommend it.", "negative"),
        ("The device has a 15-inch screen and weighs 1.4 kilograms.", "neutral"),
    ]

    for sentence, expected in test_sentences:
        features = vectorizer.transform([clean_text(sentence)])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        scores = dict(zip(model.classes_, probabilities))
        status = "PASS" if prediction == expected else "CHECK"
        print(f"\n[{status}] {sentence}")
        print(f"Expected: {expected} | Predicted: {prediction}")
        print(
            f"Scores: negative={scores.get('negative', 0):.3f}, "
            f"neutral={scores.get('neutral', 0):.3f}, "
            f"positive={scores.get('positive', 0):.3f}"
        )


if __name__ == "__main__":
    main()

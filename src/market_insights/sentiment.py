"""Template-generated feedback, random holdout and unseen-template diagnostic."""

import re
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLASSES = ["positive", "negative", "neutral"]


def clean_text(text: str) -> str:
    # Keep accents and negation; removing 'no' would erase important sentiment.
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def evaluate_sentiment(feedback):
    df = feedback.copy()
    df["clean_text"] = df.review_text.map(clean_text)
    if (
        not df.feedback_id.is_unique
        or df[["review_text", "sentiment_label"]].isna().any().any()
    ):
        raise ValueError("Feedback IDs must be unique and text/labels complete.")
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df.sentiment_label, random_state=42
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    group_train, group_test = next(
        splitter.split(df, df.sentiment_label, df.template_id)
    )
    grouped_train, grouped_test = df.index[group_train], df.index[group_test]
    if set(df.loc[grouped_train, "template_id"]) & set(
        df.loc[grouped_test, "template_id"]
    ):
        raise AssertionError("Template overlap in the group-held-out split.")
    rows = []
    heldout = None
    final_model = None
    for split, a, b in [
        ("random_stratified", train_idx, test_idx),
        ("unseen_templates", grouped_train, grouped_test),
    ]:
        for name, ngram, estimator in [
            (
                "logreg_unigrams",
                (1, 1),
                LogisticRegression(max_iter=1000, random_state=42),
            ),
            (
                "logreg_bigrams",
                (1, 2),
                LogisticRegression(max_iter=1000, random_state=42),
            ),
            ("naive_bayes", (1, 2), MultinomialNB()),
        ]:
            model = make_pipeline(
                TfidfVectorizer(max_features=3000, ngram_range=ngram, min_df=2),
                estimator,
            )
            model.fit(df.loc[a, "clean_text"], df.loc[a, "sentiment_label"])
            predictions = model.predict(df.loc[b, "clean_text"])
            y = df.loc[b, "sentiment_label"]
            per_class = f1_score(
                y, predictions, labels=CLASSES, average=None, zero_division=0
            )
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "train_n": len(a),
                    "test_n": len(b),
                    "accuracy": accuracy_score(y, predictions),
                    "macro_f1": f1_score(
                        y, predictions, average="macro", zero_division=0
                    ),
                    **{f"f1_{c}": float(v) for c, v in zip(CLASSES, per_class)},
                }
            )
            if split == "unseen_templates" and name == "logreg_bigrams":
                heldout = df.loc[
                    b, ["feedback_id", "review_text", "sentiment_label", "template_id"]
                ].copy()
                heldout["predicted"] = predictions
                final_model = model
    matrix = confusion_matrix(
        heldout.sentiment_label, heldout.predicted, labels=CLASSES
    )
    return pd.DataFrame(rows), heldout, matrix, final_model

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"subject:", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset():
    df = pd.read_csv("../data/enron_mails.csv").dropna(subset=["Message"])

    # combine Subject + Message
    df["full_text"] = (df["Subject"].fillna("") + " " + df["Message"].fillna(""))

    # clean text
    df["full_text"] = df["full_text"].apply(clean_text)

    # map spam/ham to numeric
    df["label_num"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    return df


def vectorize_data(X_train, X_test, max_features=5000, min_df=2, max_df=0.95):
    vectorizer = CountVectorizer(
        max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=(1, 2)
    )
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    return X_train_bow, X_test_bow


def fit_model(X_train_bow, y_train, alpha=0.5):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train_bow, y_train)
    return clf


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Spam): {prec:.4f}")
    print(f"Recall (Spam):    {rec:.4f}")
    print(f"F1-Score (Spam):  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


def train_and_evaluate():
    df = load_dataset()

    X = df["full_text"]
    y = df["label_num"]

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n==============================")
    print("MODEL: NAIVE BAYES WITH BIGRAMS + FIXED THRESHOLD")
    print("==============================")

    X_train_bow, X_test_bow = vectorize_data(X_train, X_test)

    clf = fit_model(X_train_bow, y_train)

    FIXED_THRESHOLD = 0.10
    y_prob = clf.predict_proba(X_test_bow)[:, 1]
    y_pred = (y_prob >= FIXED_THRESHOLD).astype(int)
    print(f"Using fixed spam threshold: {FIXED_THRESHOLD:.2f}")

    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    train_and_evaluate()
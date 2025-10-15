import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


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
    df["full_text"] = (df["Subject"].fillna("") + " " + df["Message"].fillna("")).apply(clean_text)
    df["label_num"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    return df


class SimpleBoW:
    def __init__(self):
        self.vocab = {}

    def fit(self, documents):
        vocab_set = set()
        for doc in documents:
            for word in doc.split():
                vocab_set.add(word)
        self.vocab = {word: i for i, word in enumerate(sorted(vocab_set))}

    def transform(self, documents):
        n_docs = len(documents)
        n_vocab = len(self.vocab)
        matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for i, doc in enumerate(documents):
            for word in doc.split():
                if word in self.vocab:
                    matrix[i, self.vocab[word]] += 1
        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class SimpleNaiveBayes:
    def __init__(self):
        self.class_word_probs = {}
        self.class_priors = {}

    def fit(self, X, y):
        n_docs, n_words = X.shape
        classes = np.unique(y)

        for c in classes:
            X_c = X[y == c]
            word_counts = X_c.sum(axis=0)
            total_words = word_counts.sum()

            # Laplace smoothing
            probs = (word_counts + 1) / (total_words + n_words)
            self.class_word_probs[c] = probs
            self.class_priors[c] = X_c.shape[0] / n_docs

    def predict(self, X):
        results = []
        for x in X:
            scores = {}
            for c in self.class_word_probs:
                log_prob = np.log(self.class_priors[c])
                log_prob += np.sum(x * np.log(self.class_word_probs[c]))
                scores[c] = log_prob
            pred = max(scores, key=scores.get)
            results.append(pred)
        return np.array(results)


def apply_smote(X_bow, y):
    print(f"\nOriginal class distribution: {Counter(y)}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bow, y)
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def train_and_evaluate(use_smote=False):
    df = load_dataset()
    X = df["full_text"]
    y = df["label_num"].to_numpy()

    # Split 80/20 on same dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining on 80% and testing on 20% of the same dataset.")

    bow = SimpleBoW()
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)

    if use_smote:
        X_train_bow, y_train = apply_smote(X_train_bow, y_train)

    nb = SimpleNaiveBayes()
    nb.fit(X_train_bow, y_train)

    y_pred = nb.predict(X_test_bow)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    print("=" * 50)
    print("WITHOUT SMOTE")
    print("=" * 50)
    train_and_evaluate(use_smote=False)

    print("\n" + "=" * 50)
    print("WITH SMOTE")
    print("=" * 50)
    train_and_evaluate(use_smote=True)
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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


def load_datasets():
    df_enron = pd.read_csv("../data/enron_mails.csv").dropna(subset=["Message"])
    df_venky = pd.read_csv("../data/venky_spam_ham_dataset.csv").dropna(subset=["text"])

    df_enron["full_text"] = (df_enron["Subject"].fillna("") + " " + df_enron["Message"].fillna(""))
    df_enron["full_text"] = df_enron["full_text"].apply(clean_text)
    df_venky["text"] = df_venky["text"].apply(clean_text)

    df_enron["label_num"] = df_enron["Spam/Ham"].map({"ham": 0, "spam": 1})

    return df_enron, df_venky


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


def train_and_evaluate(train_enron=1):
    df_enron, df_venky = load_datasets()

    if train_enron == 1:
        X_train = df_enron["full_text"]
        y_train = df_enron["label_num"]
        X_test = df_venky["text"]
        y_test = df_venky["label_num"]
        print("\nTraining on Enron, Testing on Venky")
    else:
        X_train = df_venky["text"]
        y_train = df_venky["label_num"]
        X_test = df_enron["full_text"]
        y_test = df_enron["label_num"]
        print("\nTraining on Venky, Testing on Enron")

    X_train_bow, X_test_bow = vectorize_data(X_train, X_test)

    clf = fit_model(X_train_bow, y_train)

    FIXED_THRESHOLD = 0.10
    y_prob = clf.predict_proba(X_test_bow)[:, 1]
    y_pred = (y_prob >= FIXED_THRESHOLD).astype(int)
    print(f"Using fixed spam threshold: {FIXED_THRESHOLD:.2f}")

    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    train_and_evaluate(train_enron=0)

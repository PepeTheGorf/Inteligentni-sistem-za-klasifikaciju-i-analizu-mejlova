import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

#formatting text
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

#preparing dataset
def load_datasets():
    df_enron = pd.read_csv("../data/enron_mails.csv").dropna(subset=["Message"])
    df_venky = pd.read_csv("../data/venky_spam_ham_dataset.csv").dropna(subset=["text"])

    df_enron["full_text"] = (
        df_enron["Subject"].fillna("") + " " + df_enron["Message"].fillna("")
    ).apply(clean_text)
    df_venky["text"] = df_venky["text"].apply(clean_text)

    df_enron["label_num"] = df_enron["Spam/Ham"].map({"ham": 0, "spam": 1})
    df_venky["label_num"] = df_venky["label"].map({"ham": 0, "spam": 1})

    return df_enron, df_venky

def apply_smote(X_bow, y):
    print(f"\nOriginal class distribution: {Counter(y)}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bow, y)

    print(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled


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
            #getting documents of class
            X_c = X[y == c]
            word_counts = X_c.sum(axis=0)
            total_words = word_counts.sum()

            #laplace smoothing
            probs = (word_counts + 1) / (total_words + n_words) #likelihood of word appearing in class
            self.class_word_probs[c] = probs
            self.class_priors[c] = X_c.shape[0] / n_docs    #likelihood of being that class

    def predict(self, X):
        results = []
        #for document
        for x in X:
            scores = {}
            #for each class
            for c in self.class_word_probs:
                log_prob = np.log(self.class_priors[c])
                log_prob += np.sum(x * np.log(self.class_word_probs[c]))
                scores[c] = log_prob
            pred = max(scores, key=scores.get)
            results.append(pred)
        return np.array(results)

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def train_and_evaluate(train_enron=1, use_smote=False, mode="cross"):
    df_enron, df_venky = load_datasets()

    if mode == "cross":
        if train_enron == 1:
            X_train = df_enron["full_text"]
            y_train = df_enron["label_num"].to_numpy()
            X_test = df_venky["text"]
            y_test = df_venky["label_num"].to_numpy()
            print("\nTraining on Enron, Testing on Venky")
        else:
            X_train = df_venky["text"]
            y_train = df_venky["label_num"].to_numpy()
            X_test = df_enron["full_text"]
            y_test = df_enron["label_num"].to_numpy()
            print("\nTraining on Venky, Testing on Enron")

    elif mode == "split":
        from sklearn.model_selection import train_test_split
        #enron 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            df_enron["full_text"], df_enron["label_num"].to_numpy(),
            test_size=0.2, random_state=42, stratify=df_enron["label_num"]
        )
        print("\nTraining on 80% and testing on 20% of Enron")
    else:
        raise ValueError("mode must be 'cross' or 'split'")

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
    print("Cross WITHOUT SMOTE")
    print("=" * 50)
    train_and_evaluate(train_enron=0, use_smote=False)

    print("\n" + "=" * 50)
    print("Cross WITH SMOTE")
    print("=" * 50)
    train_and_evaluate(train_enron=0, use_smote=True)

    print("=" * 50)
    print("Standard WITHOUT SMOTE")
    print("=" * 50)
    train_and_evaluate(train_enron=0, use_smote=False)

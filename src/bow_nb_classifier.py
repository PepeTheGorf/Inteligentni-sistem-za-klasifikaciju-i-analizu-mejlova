import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df_train = pd.read_csv('../data/enron_spam_data.csv')
df_test = pd.read_csv('../data/venky_spam_ham_dataset.csv')

df_train = df_train.dropna(subset=['Message'])
df_test = df_test.dropna(subset=['text'])

X_train = df_train['Message']
y_train = df_train['Spam/Ham'].map({'ham': 0, 'spam': 1})

X_test = df_test['text']
y_test = df_test['label_num']

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

y_pred = clf.predict(X_test_bow)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pandas as pd

# Load your dataset
df = pd.read_csv("emails_cleaned.csv")

print(df['Spam/Ham'].value_counts())



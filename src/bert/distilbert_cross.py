import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, \
    recall_score, confusion_matrix
import warnings
#from google.colab import drive

warnings.filterwarnings('ignore')

os.environ["WANDB_DISABLED"] = "true"

#drive.mount('/content/drive')

def prepare_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    train_dict = {'text': X_train.tolist(), 'label': y_train.tolist()}
    val_dict = {'text': X_val.tolist(), 'label': y_val.tolist()}
    test_dict = {'text': X_test.tolist(), 'label': y_test.tolist()}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    return train_dataset, val_dataset, test_dataset


def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    spam_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'spam_recall': spam_recall
    }

def train_with_trainer(X_train, y_train, X_test, y_test):
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01

    print(f"\nHyperparameters:")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(f"\n{'=' * 60}")
    print("DATASET SPLITS")
    print(f"{'=' * 60}")
    print(f"Train: {len(X_train_split)} (Spam: {sum(y_train_split)})")
    print(f"Val: {len(X_val_split)} (Spam: {sum(y_val_split)})")
    print(f"Test (Venky): {len(X_test)} (Spam: {sum(y_test)})")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        X_test, y_test
    )

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\n" + "=" * 60)
    print("TRAINING ON ENRON")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    val_results = trainer.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"Validation F1: {val_results['eval_f1']:.4f}")
    print(f"Validation Spam Recall: {val_results['eval_spam_recall']:.4f}")

    print("\n" + "=" * 60)
    print("TESTING ON VENKY DATASET")
    print("=" * 60)

    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = (test_dataset['label'])

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=['ham', 'spam'],
        zero_division=0
    ))

    test_acc = accuracy_score(true_labels, predicted_labels)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', pos_label=1, zero_division=0
    )
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"\nTest Metrics Summary:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision (spam): {test_prec:.4f}")
    print(f"Recall (spam): {test_rec:.4f}")
    print(f"F1 (spam): {test_f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    print("\nSaving model...")
    model.save_pretrained('./fine_tuned_spam_classifier')
    tokenizer.save_pretrained('./fine_tuned_spam_classifier')

    return model, tokenizer, trainer


def main():
    df_enron = pd.read_csv("/content/drive/MyDrive/EmailDatasets/enron_mails.csv").dropna(subset=['Message'])
    X_enron = df_enron['Message'].values
    y_enron = df_enron['Spam/Ham'].map({'ham': 0, 'spam': 1}).values

    df_venky = pd.read_csv("/content/drive/MyDrive/EmailDatasets/venky_spam_ham_dataset.csv").dropna(subset=['text'])
    X_venky = df_venky['text'].values
    y_venky = df_venky['label'].map({'ham': 0, 'spam': 1}).values

    model, tokenizer, trainer = train_with_trainer(X_enron, y_enron, X_venky, y_venky)
    return model, tokenizer, trainer

if __name__ == "__main__":
    main()
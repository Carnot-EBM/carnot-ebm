import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure we can load tier0e
import sys
import os
sys.path.insert(0, os.path.abspath('python'))

try:
    from carnot.verify.tier0e_eorm import EORMVerifier
    tier0e_exists = True
except ImportError:
    tier0e_exists = False

def compute_ece(predictions, labels, n_bins=10):
    predictions = np.array(predictions)
    labels = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

texts = []
labels = []
with open("data/fover_corpus.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["step_text"])
        labels.append(1 if data["label"] == "correct" else 0)

n_corpus_pairs = len(texts)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

if tier0e_exists:
    verifier = EORMVerifier()
    vectorizer = verifier.vectorizer
else:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_text)

X_train_features = vectorizer.transform(X_train_text)
X_test_features = vectorizer.transform(X_test_text)

if tier0e_exists:
    baseline_probs = verifier.clf.predict_proba(X_test_features)[:, 1]
else:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_features, y_train)
    baseline_probs = clf.predict_proba(X_test_features)[:, 1]

baseline_ece = compute_ece(baseline_probs, y_test)
baseline_auroc = roc_auc_score(y_test, baseline_probs)

print(f"n_corpus_pairs: {n_corpus_pairs}")
print(f"baseline_ece: {baseline_ece}")
print(f"baseline_auroc: {baseline_auroc}")

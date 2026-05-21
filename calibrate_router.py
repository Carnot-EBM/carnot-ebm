import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load data
data = []
with open("data/fover_corpus.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

texts = [d["step_text"] for d in data]
y = np.array([1 if d["label"] == "incorrect" else 0 for d in data])

# 2. Split
texts_train, texts_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# 3. Train proxy model
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict weak_score (probability of incorrect)
scores_test = model.predict_proba(X_test)[:, 1]

# 4. Calibrate t_low (FNR <= 0.05)
# FNR = sum(scores < t & y == 1) / sum(y == 1)
t_low = 0.0
for t in np.linspace(0.0, 1.0, 1001):
    fnr = np.sum((scores_test < t) & (y_test == 1)) / np.sum(y_test == 1)
    if fnr > 0.05:
        break
    t_low = t

# 5. Calibrate t_high (FPR <= 0.10)
# FPR = sum(scores > t & y == 0) / sum(y == 0)
t_high = 1.0
for t in np.linspace(1.0, 0.0, 1001):
    fpr = np.sum((scores_test > t) & (y_test == 0)) / np.sum(y_test == 0)
    if fpr > 0.10:
        break
    t_high = t

print(f"t_low={t_low:.4f}, t_high={t_high:.4f}")

# 6. Evaluate on 100 examples from eval split (test split)
# The instructions say: Evaluate on FoVer eval split (N=100 examples, random_seed=42)
np.random.seed(42)
indices = np.random.choice(len(y_test), 100, replace=False)
scores_100 = scores_test[indices]
y_100 = y_test[indices]

n_accepted_early = 0
n_full_ensemble = 0
n_partial_verify = 0
for score in scores_100:
    if score < t_low:
        n_accepted_early += 1
    elif score > t_high:
        n_full_ensemble += 1
    else:
        n_partial_verify += 1

policy_savings_pct = (n_accepted_early + n_partial_verify * 0.5) / 100.0 * 100.0

accepted_indices = (scores_100 < t_low)
total_incorrect = np.sum(y_100 == 1)
false_negative_rate = 0.0
if total_incorrect > 0:
    false_negative_rate = np.sum((scores_100 < t_low) & (y_100 == 1)) / total_incorrect

print(f"n_accepted_early={n_accepted_early}")
print(f"n_partial_verify={n_partial_verify}")
print(f"n_full_ensemble={n_full_ensemble}")
print(f"policy_savings_pct={policy_savings_pct}")
print(f"false_negative_rate={false_negative_rate:.4f}")

# Save the model and vectorizer for the WeakStrongRouter
import pickle
with open("results/weak_strong_proxy.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "model": model}, f)


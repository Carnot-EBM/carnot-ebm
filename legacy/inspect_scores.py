import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split

with open("results/weak_strong_proxy.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
vectorizer = data["vectorizer"]

corpus = []
with open("data/fover_corpus.jsonl") as f:
    for line in f:
        corpus.append(json.loads(line))

texts = [d["step_text"] for d in corpus]
# What if it was trained with 1 = correct?
y_correct = np.array([1 if d["label"] == "correct" else 0 for d in corpus])
y_incorrect = np.array([1 if d["label"] == "incorrect" else 0 for d in corpus])

X = vectorizer.transform(texts)
scores = model.predict_proba(X)[:, 1]

print("Mean score for label=correct:", np.mean(scores[y_correct == 1]))
print("Mean score for label=incorrect:", np.mean(scores[y_incorrect == 1]))

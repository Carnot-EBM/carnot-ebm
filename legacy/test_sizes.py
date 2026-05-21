import json
import numpy as np
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

corpus_texts = []
labels = []
with open("data/fover_corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        corpus_texts.append(d.get("step_text", ""))
        labels.append(1 if d.get("label") == "incorrect" else 0)

vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(corpus_texts)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
cluster_labels = dbscan.fit_predict(X)
sizes = Counter(cluster_labels)

correct_sizes = []
incorrect_sizes = []
for i, l in enumerate(labels):
    s = sizes[cluster_labels[i]]
    if l == 0:
        correct_sizes.append(s)
    else:
        incorrect_sizes.append(s)

print("Avg cluster size correct:", np.mean(correct_sizes))
print("Avg cluster size incorrect:", np.mean(incorrect_sizes))

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

corpus_texts = []
labels = []
with open("data/fover_corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        corpus_texts.append(str(d.get("question_id", "")) + " " + d.get("step_text", ""))
        labels.append(1 if d.get("label") == "incorrect" else 0)

vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(corpus_texts)
nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X)

energies = []
for i in range(len(corpus_texts)):
    distances, _ = nn.kneighbors(X[i])
    cluster_size = np.sum(distances[0] < 0.3)
    if cluster_size == 0:
        cluster_size = 1
    energies.append(-np.log(cluster_size / len(corpus_texts) + 1e-9))

print("AUROC:", roc_auc_score(labels, energies))

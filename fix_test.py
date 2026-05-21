import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# 1. Load data
data = []
with open("data/fover_corpus.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

texts = [d["step_text"] for d in data]
# The prompt says: "Fix: set y_correct = 0 (incorrect is class=1) in the calibration fitting step"
# But it's ALREADY 1 if incorrect in the proxy. What if the energy is NOT from the proxy, but from ODAR?
# In WeakStrongRouter:
# complexity = len(prompt.split()) / 100.0
# weak_score = complexity - 0.5

import numpy as np
from sklearn.metrics import roc_auc_score
labels = [0, 0, 1, 1]
# 0 has lower scores
scores = [0.1, 0.2, 0.8, 0.9]
print("Expected > 0.5:", roc_auc_score(labels, scores))

# 0 has higher scores
scores = [0.8, 0.9, 0.1, 0.2]
print("Expected < 0.5:", roc_auc_score(labels, scores))

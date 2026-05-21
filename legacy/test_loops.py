import numpy as np

# Perfect model:
scores_test = np.array([0.0]*50 + [1.0]*50)  # 0.0 for correct, 1.0 for incorrect
y_test = np.array([0]*50 + [1]*50)

t_low = 0.0
for t in np.linspace(0.0, 1.0, 1001):
    fnr = np.sum((scores_test < t) & (y_test == 1)) / np.sum(y_test == 1)
    if fnr > 0.05:
        break
    t_low = t

t_high = 1.0
for t in np.linspace(1.0, 0.0, 1001):
    fpr = np.sum((scores_test > t) & (y_test == 0)) / np.sum(y_test == 0)
    if fpr > 0.10:
        break
    t_high = t

print(t_low, t_high)

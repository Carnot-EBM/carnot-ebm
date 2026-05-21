import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def test_set(statements):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
    try:
        X = vec.fit_transform(statements)
    except ValueError:
        return 1.0
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 1.0)
    min_sim = np.min(sim)
    print(f"{statements} -> {1.0 - min_sim}")

test_set(["The sum of 2+3 is 5", "The sum of 2+3 is 7"])
test_set(["The cat is alive", "The cat is dead"])
test_set(["X > Y", "X < Y"])

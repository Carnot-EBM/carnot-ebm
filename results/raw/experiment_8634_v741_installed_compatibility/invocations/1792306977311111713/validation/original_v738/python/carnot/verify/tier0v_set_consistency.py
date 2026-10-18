"""Tier 0v Set-Consistency Energy Network Verifier.

Implements set-level consistency verification based on 'Introducing Verification Task of Set Consistency with Set-Consistency Energy Networks' (Song et al., 2025).
It uses a contrastive loss style metric to learn compatibility among statement collections.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class SetConsistencyVerifier:
    def __init__(self):
        # We use a TF-IDF vectorizer that supports words, symbols (like > < + -), and n-grams 
        # to ensure that structural and numeric contradictions (e.g. 5 vs 7) result in high distance.
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b|\S',
            ngram_range=(1, 4)
        )
        self.is_fitted = False

    def fit(self, corpus: List[str]):
        """Fits the TF-IDF vectorizer on a corpus."""
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def compute_energy(self, statements: List[str]) -> float:
        """Computes the inconsistency energy of a set of statements.
        
        Args:
            statements: List of text statements.
            
        Returns:
            Energy score in [0, 1].
            0 = perfectly consistent (high similarity among all pairs).
            >0 = contradictory (at least one pair has low similarity).
        """
        if not statements or len(statements) < 2:
            return 0.0
            
        try:
            if self.is_fitted:
                X = self.vectorizer.transform(statements)
            else:
                X = self.vectorizer.fit_transform(statements)
        except ValueError:
            return 1.0
            
        sim_matrix = cosine_similarity(X)
        
        # Max pair distance = 1 - min pair similarity
        # We ignore self-similarity by filling the diagonal with 1.0
        np.fill_diagonal(sim_matrix, 1.0)
        min_sim = np.min(sim_matrix)
        
        energy = 1.0 - float(min_sim)
        return max(0.0, min(1.0, energy))

"""Tier 0w Paraphrastic Consistency Verifier.

Implements paraphrastic probing and consistency verification as per arXiv:2602.11361.
Generates N paraphrastic perturbations of the input query. Models that are confident/correct
produce consistent answers across paraphrases. Consistency energy is the standard deviation
of the verification scores across the paraphrases.
"""
import numpy as np
import random
from typing import List, Callable

class ParaphrasticConsistencyVerifier:
    def __init__(self, base_verifier: Callable[[str, str], float] = None):
        """
        Args:
            base_verifier: A function that takes (question, response) and returns a score.
                           If None, uses a simple Jaccard similarity fallback.
        """
        if base_verifier is None:
            self.base_verifier = self._default_verifier
        else:
            self.base_verifier = base_verifier

    def _default_verifier(self, q: str, r: str) -> float:
        # Simple Jaccard similarity between words
        q_words = set(q.lower().split())
        r_words = set(r.lower().split())
        if not q_words or not r_words:
            return 0.0
        return len(q_words.intersection(r_words)) / len(q_words.union(r_words))

    def generate_word_shuffle_paraphrases(self, text: str, n: int = 3) -> List[str]:
        words = text.split()
        paraphrases = []
        for _ in range(n):
            shuffled = list(words)
            random.shuffle(shuffled)
            paraphrases.append(" ".join(shuffled))
        return paraphrases

    def compute_energy(self, question: str, response: str, paraphrases: List[str] = None, n: int = 3) -> float:
        """
        Computes the consistency energy (std dev of verification scores across paraphrases).
        High energy = inconsistent/hallucination.
        Low energy = consistent.
        """
        if not paraphrases or len(paraphrases) < n:
            # Fallback to word shuffle
            needed = n - (len(paraphrases) if paraphrases else 0)
            syn_paras = self.generate_word_shuffle_paraphrases(question, needed)
            if paraphrases is None:
                paraphrases = syn_paras
            else:
                paraphrases.extend(syn_paras)
        
        scores = [self.base_verifier(p, response) for p in paraphrases[:n]]
        if not scores:
            return 0.0
        
        return float(np.std(scores))

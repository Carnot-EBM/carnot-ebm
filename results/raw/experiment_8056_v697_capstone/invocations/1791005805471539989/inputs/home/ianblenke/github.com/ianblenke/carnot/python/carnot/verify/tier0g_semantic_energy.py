import json
import numpy as np
import os
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

class SemanticEnergyVerifier:
    def __init__(self, corpus_path="data/fover_corpus.jsonl", max_features=5000, random_seed=42):
        self.corpus_path = corpus_path
        self.vectorizer = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(3,5))
        self.corpus_texts = []
        
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("step_text", "")
                    self.corpus_texts.append(text)
        
        self.n_corpus_entries = len(self.corpus_texts)
        if self.n_corpus_entries > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)
            
            dbscan = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
            self.labels = dbscan.fit_predict(self.tfidf_matrix)
            self.n_clusters = len(set(self.labels))
            
            os.makedirs("results", exist_ok=True)
            with open("results/tier0g_cluster_index.pkl", "wb") as f:
                pickle.dump(self.labels, f)
                
            self.cluster_sizes = Counter(self.labels)
            
            self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.nn.fit(self.tfidf_matrix)
        else:
            self.n_clusters = 0

    def compute_energy(self, question, response, epsilon=1e-9):
        text = str(response)
        if not hasattr(self, 'tfidf_matrix'):
            return 0.0
            
        vec = self.vectorizer.transform([text])
        distances, indices = self.nn.kneighbors(vec)
        
        min_dist = distances[0][0]
        best_idx = indices[0][0]
        
        if min_dist < 0.3:
            cluster_id = self.labels[best_idx]
            cluster_size = self.cluster_sizes[cluster_id]
        else:
            cluster_size = 1
            
        energy = -np.log(cluster_size / self.n_corpus_entries + epsilon)
        return float(energy)

    def verify(self, question, response):
        energy = self.compute_energy(question, response)
        # Using a fixed threshold or just returning the energy
        # For typical verifiers in this repo, it might return a VerificationResult
        # But we can just use compute_energy for the experiment.
        return energy

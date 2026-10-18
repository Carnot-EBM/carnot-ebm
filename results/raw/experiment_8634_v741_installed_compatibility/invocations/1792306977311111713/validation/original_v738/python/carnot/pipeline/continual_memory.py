import numpy as np
from typing import List, Dict, Any

class ContinualMemory:
    def __init__(self):
        self.memory_states: List[Dict[str, Any]] = []
    
    def add_state(self, vector: np.ndarray, metadata: Dict[str, Any]):
        self.memory_states.append({"vector": vector, "metadata": metadata})
        
    def distill(self, n_clusters: int) -> None:
        if len(self.memory_states) <= n_clusters:
            return
            
        vectors = np.array([state["vector"] for state in self.memory_states])
        
        np.random.seed(42)
        indices = np.random.choice(len(vectors), n_clusters, replace=False)
        centroids = vectors[indices]
        
        for _ in range(10):
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = []
            for i in range(n_clusters):
                cluster_pts = vectors[labels == i]
                if len(cluster_pts) > 0:
                    new_centroids.append(cluster_pts.mean(axis=0))
                else:
                    new_centroids.append(centroids[i])
            centroids = np.array(new_centroids)
            
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_states = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue
            
            centroid = centroids[i]
            cluster_vectors = vectors[cluster_indices]
            dists_to_centroid = np.linalg.norm(cluster_vectors - centroid, axis=1)
            best_idx = cluster_indices[np.argmin(dists_to_centroid)]
            new_states.append(self.memory_states[best_idx])
            
        self.memory_states = new_states

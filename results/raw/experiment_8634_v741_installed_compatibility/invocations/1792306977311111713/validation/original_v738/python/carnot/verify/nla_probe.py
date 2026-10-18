import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalSAE(nn.Module):
    """Tiny Sparse Autoencoder for NLA-class probing."""
    def __init__(self, d_model: int, expansion_factor: int = 4, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        
        self.encoder = nn.Linear(d_model, self.d_sae, bias=True)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(self.d_sae, d_model, bias=True)
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.encoder.weight.dtype)
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.encoder.weight.dtype)
        with torch.no_grad():
            decoded, _ = self(x)
            return torch.nn.functional.mse_loss(decoded, x, reduction='none').mean(dim=-1)


class NLAClassProbe:
    """
    NLA-Class Probing as the 16th Verifier.
    Uses SAE-based reconstruction error on target LLM activations to detect adversarial outputs.
    """
    def __init__(self, d_model: int = 4096, expansion_factor: int = 4, device: str = 'cpu'):
        self.device = device
        self.d_model = d_model
        # Use custom SAE as v0 baseline per requirements to meet time budget.
        self.sae = MinimalSAE(d_model=d_model, expansion_factor=expansion_factor).to(device)
        
    def train_step(self, activations: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        activations = activations.to(self.sae.encoder.weight.dtype)
        self.sae.train()
        optimizer.zero_grad()
        decoded, encoded = self.sae(activations)
        reconstruction_loss = torch.nn.functional.mse_loss(decoded, activations)
        l1_loss = self.sae.l1_coeff * encoded.abs().sum(dim=-1).mean()
        loss = reconstruction_loss + l1_loss
        loss.backward()
        optimizer.step()
        
        # normalize decoder weights
        with torch.no_grad():
            self.sae.decoder.weight.data = self.sae.decoder.weight.data / self.sae.decoder.weight.data.norm(dim=0, keepdim=True)
        return loss.item()

    def score(self, prompt: str, candidate: str, activations: torch.Tensor) -> float:
        """
        Returns a confidence score in [0, 1].
        High reconstruction error -> low confidence.
        """
        self.sae.eval()
        activations = activations.to(self.sae.encoder.weight.dtype)
        with torch.no_grad():
            mse = self.sae.reconstruction_error(activations)
            mean_mse = mse.mean().item()
            confidence = 1.0 / (1.0 + mean_mse)
            return confidence

    def feature_description_collision_rate(
        self,
        activations: torch.Tensor,
        cosine_threshold: float = 0.95,
    ) -> dict:
        """
        Measure the fraction of SAE feature-dictionary pairs whose decoder
        vectors are so similar that an auto-interpretability system would
        assign them near-identical descriptions.

        WHY this matters: SAEs can learn redundant features — two dictionary
        atoms that encode the same direction in activation space.  When a
        downstream labeller (human or LLM) auto-interprets features, redundant
        pairs receive collision descriptions like "activates on nouns" vs
        "activates on nouns (variant)".  A high collision rate signals that the
        SAE is wasting capacity on repetition rather than learning a diverse
        semantic basis, which undermines NLA-class probing reliability.

        Implementation: we use pairwise cosine similarity between the SAE
        decoder's column vectors (each column = one feature direction) as a
        cheap, label-free proxy for description similarity.  Two features with
        cosine similarity ≥ `cosine_threshold` are a "collision pair" —
        they point in nearly the same direction so any description that fits
        one would fit the other.

        Args:
            activations: Tensor of shape (n_samples, d_model) drawn from the
                         target model (or synthetic stand-in).  Used to
                         optionally fine-train the SAE before measuring; if the
                         SAE is already trained, you can pass a single dummy
                         row.
            cosine_threshold: Similarity cutoff above which two feature vectors
                              count as a collision.  Default 0.95 (near-parallel
                              directions that a description system cannot
                              distinguish).

        Returns:
            dict with keys:
              collision_rate  – fraction of unique feature pairs that collide
              n_features      – total number of SAE dictionary features
              n_collision_pairs – count of colliding pairs
              n_total_pairs   – total unique pairs considered
              cosine_threshold – the threshold used
        """
        self.sae.eval()
        with torch.no_grad():
            # decoder weight is (d_model, d_sae) → each column = one feature direction
            W = self.sae.decoder.weight  # shape: (d_model, d_sae)
            # L2-normalise each column so cosine similarity = dot product
            W_norm = F.normalize(W, dim=0)  # normalise along d_model axis
            # pairwise cosine similarity matrix: (d_sae, d_sae)
            sim = W_norm.T @ W_norm
            # upper-triangle mask to count each pair once (exclude self-similarity diagonal)
            d_sae = sim.shape[0]
            triu_mask = torch.triu(torch.ones(d_sae, d_sae, dtype=torch.bool), diagonal=1)
            upper_sim = sim[triu_mask]
            n_total_pairs = upper_sim.numel()
            n_collision_pairs = int((upper_sim >= cosine_threshold).sum().item())
            collision_rate = n_collision_pairs / max(n_total_pairs, 1)
        return {
            "collision_rate": collision_rate,
            "n_features": d_sae,
            "n_collision_pairs": n_collision_pairs,
            "n_total_pairs": n_total_pairs,
            "cosine_threshold": cosine_threshold,
        }

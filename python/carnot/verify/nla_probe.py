import torch
import torch.nn as nn

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

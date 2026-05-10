"""Online SGD/AdamW updater for CIKAN models.

Spec: REQ-LEARN-101
"""

from typing import Any, Sequence
import numpy as np
from carnot.models.cikan_verifier import CIKAN

class OnlineUpdater:
    """Online fine-tuning of the CIKAN residual KAN head on streaming violations."""
    
    def __init__(
        self, 
        optimizer: str = "adamw", 
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ):
        if optimizer not in ("sgd", "adamw"):
            raise ValueError(f"Unknown optimizer: {optimizer}")
            
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.m_ctrl: np.ndarray | None = None
        self.v_ctrl: np.ndarray | None = None
        self.m_bias = 0.0
        self.v_bias = 0.0
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def step(self, cikan: CIKAN, x: Sequence[float], y: float) -> float:
        """Perform one online gradient step on the CIKAN residual head.
        
        y=1.0 means valid (satisfies constraints), y=0.0 means violation.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        
        if self.m_ctrl is None:
            self.m_ctrl = np.zeros_like(cikan.residual_control_points)
            self.v_ctrl = np.zeros_like(cikan.residual_control_points)
            
        self.t += 1
        
        energy = cikan.energy(x_arr)
        prob = 1.0 / (1.0 + np.exp(energy))
        
        d_loss_d_energy = float(y - prob)
        
        grad_ctrl = np.zeros_like(cikan.residual_control_points)
        grad_bias = d_loss_d_energy
        
        for feature_idx, value in enumerate(x_arr):
            grad_ctrl[feature_idx] += d_loss_d_energy * cikan._basis_values(float(value))
            
        if self.optimizer == "sgd":
            cikan.bias -= self.learning_rate * grad_bias
            cikan.residual_control_points -= self.learning_rate * grad_ctrl
        else:
            # adamw
            self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * grad_bias
            self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (grad_bias ** 2)
            
            m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
            v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)
            
            cikan.bias -= self.learning_rate * (m_hat_bias / (np.sqrt(v_hat_bias) + self.eps) + self.weight_decay * cikan.bias)
            
            self.m_ctrl = self.beta1 * self.m_ctrl + (1 - self.beta1) * grad_ctrl
            self.v_ctrl = self.beta2 * self.v_ctrl + (1 - self.beta2) * (grad_ctrl ** 2)
            
            m_hat_ctrl = self.m_ctrl / (1 - self.beta1 ** self.t)
            v_hat_ctrl = self.v_ctrl / (1 - self.beta2 ** self.t)
            
            cikan.residual_control_points -= self.learning_rate * (m_hat_ctrl / (np.sqrt(v_hat_ctrl) + self.eps) + self.weight_decay * cikan.residual_control_points)
            
        np.clip(cikan.residual_control_points, -1.0, 1.0, out=cikan.residual_control_points)
        cikan.bias = float(np.clip(cikan.bias, -1.0, 1.0))
        
        probs = np.clip(np.array([prob]), 1e-9, 1.0 - 1e-9)
        labels = np.array([y])
        loss = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
        return loss

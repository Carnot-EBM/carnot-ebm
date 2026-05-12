import jax.numpy as jnp
from carnot.verify.energy_proxy import DummyEnergyExtractionProxy
from carnot.verify.constraint import BaseConstraint

class MockValidator(BaseConstraint):
    @property
    def name(self) -> str:
        return "mock_validator"
        
    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x ** 2)

def test_dummy_energy_extraction_proxy():
    validator = MockValidator()
    proxy = DummyEnergyExtractionProxy(validator)
    
    x = jnp.array([1.0, 2.0])
    meta = proxy.extract_metadata(x)
    
    assert meta["glauber_compatible"] is True
    assert meta["diffusion_compatible"] is True
    assert meta["validator_name"] == "mock_validator"
    assert "energy_val" in meta
    assert "grad_norm" in meta
    assert meta["continuous_latent_scoring_ready"] is True
    assert meta["energy_val"] == 5.0

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
    assert "automata_metadata" in meta
    assert "validator_metadata" in meta
    assert meta["generator_integration_claim"] is False

def test_validator_metadata_roundtrip():
    import json
    validator = MockValidator()
    proxy = DummyEnergyExtractionProxy(validator)
    
    x = jnp.array([1.0, 2.0])
    meta = proxy.extract_metadata(x)
    
    # Ensure it's JSON serializable (roundtrip)
    # The jnp.float32 for energy_val and grad_norm might cause issues if not float,
    # but the proxy casts them to float(), so this should pass.
    roundtrip = json.loads(json.dumps(meta))
    assert roundtrip["validator_name"] == "mock_validator"
    assert roundtrip["generator_integration_claim"] is False
    assert roundtrip["automata_metadata"] == {}
    assert roundtrip["validator_metadata"] == {}


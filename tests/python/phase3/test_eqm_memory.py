import pytest
import jax.numpy as jnp
from carnot.phase3.eqm_memory import EqMMemoryCache

def test_eqm_memory_cache(tmp_path):
    """
    Tests REQ-KONA-2102: EqM Parameter Memory Cache.
    SCENARIO-KONA-2102: Successfully serialize and retrieve parameters.
    """
    cache = EqMMemoryCache(cache_dir=str(tmp_path))
    
    # Test with JAX Array
    theta_array = jnp.array([1.0, 2.0, 3.0])
    cache.save_parameters("test_key_array", theta_array)
    loaded_array = cache.load_parameters("test_key_array")
    assert jnp.allclose(theta_array, loaded_array)
    
    # Test with Dict
    theta_dict = {"weights": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "bias": jnp.array([0.5, 0.5])}
    cache.save_parameters("test_key_dict", theta_dict)
    loaded_dict = cache.load_parameters("test_key_dict")
    assert jnp.allclose(theta_dict["weights"], loaded_dict["weights"])
    assert jnp.allclose(theta_dict["bias"], loaded_dict["bias"])
    
    # Test with Nested Dict
    theta_nested = {"layer1": {"w": jnp.array([1.0]), "b": jnp.array([2.0])}}
    cache.save_parameters("test_key_nested", theta_nested)
    loaded_nested = cache.load_parameters("test_key_nested")
    assert jnp.allclose(theta_nested["layer1"]["w"], loaded_nested["layer1"]["w"])
    assert jnp.allclose(theta_nested["layer1"]["b"], loaded_nested["layer1"]["b"])
    
    # Test missing key
    assert cache.load_parameters("missing_key") is None

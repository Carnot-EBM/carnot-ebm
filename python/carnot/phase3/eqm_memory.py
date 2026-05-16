import json
import os
import jax
import jax.numpy as jnp
from typing import Any

def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    elif isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    else:
        return x

def _to_array(x):
    if isinstance(x, list):
        return jnp.array(x)
    elif isinstance(x, dict):
        return {k: _to_array(v) for k, v in x.items()}
    else:
        return x

class EqMMemoryCache:
    """
    Memory cache that saves and retrieves converged EqM landscapes.
    Supports basic JSON serialization of EqM parameters to enable hot-starting
    future evaluations on similar problems.
    """
    def __init__(self, cache_dir: str = "results/eqm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_filepath(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def save_parameters(self, cache_key: str, theta: Any) -> None:
        """
        Saves EqM parameters to the cache using JSON serialization.
        """
        filepath = self._get_filepath(cache_key)
        
        try:
            val = _to_list(theta)
            data = {"type": "tree", "value": val}
        except Exception:
            data = {"type": "raw", "value": theta}
                
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_parameters(self, cache_key: str) -> Any:
        """
        Retrieves parameters from the cache and hot-starts the EqM parameters
        by converting them back to JAX arrays.
        """
        filepath = self._get_filepath(cache_key)
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, "r") as f:
            data = json.load(f)
            
        t = data.get("type")
        val = data.get("value")
        
        if t == "tree":
            return _to_array(val)
        return val

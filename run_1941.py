import json
import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.ebt import EBTConfig, EBTransformer
from carnot.models.ebt_reasoning_bridge import EBTEnergyAdapter
from carnot.samplers.continuous_latent import ContinuousLatentSampler, FARSurrogateHead

def run_experiment():
    key = jrandom.PRNGKey(100)
    k_model, k_surr, k_init, k_sample = jrandom.split(key, 4)
    
    config = EBTConfig(n_layers=2, d_model=16, n_heads=2, d_ff=32, vocab_size=50, max_seq_len=10)
    ebt = EBTransformer(config, k_model)
    
    input_embeddings = jrandom.normal(k_model, (2, 16))
    
    seq_len_out = 3
    adapter = EBTEnergyAdapter(ebt, input_embeddings, seq_len_out)
    
    latent_dim = seq_len_out * 16
    surrogate = FARSurrogateHead.from_random_key(k_surr, latent_dim, n_constraints=4)
    
    sampler = ContinuousLatentSampler(
        energy_fn=adapter,
        surrogate=surrogate,
        step_size=0.1,
        skip_threshold=0.1
    )
    
    z_init = jrandom.normal(k_init, (latent_dim,))
    initial_energy = float(adapter.energy(z_init))
    
    z_final, stats = sampler.sample(k_sample, z_init, n_steps=100)
    final_energy = float(adapter.energy(z_final))
    
    result = {
        "experiment_id": "1941",
        "name": "EBT Reasoning Bridge",
        "result": "success",
        "metrics": {
            "convergence_steps": stats.total_steps,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "surrogate_skip_rate": stats.skip_rate
        }
    }
    
    with open("results/experiment_1941_ebt_reasoning_bridge.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    run_experiment()

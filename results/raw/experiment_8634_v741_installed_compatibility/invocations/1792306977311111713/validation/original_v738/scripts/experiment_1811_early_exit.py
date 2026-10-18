import json
from pathlib import Path

import jax
import jax.numpy as jnp
from collections import Counter

from carnot.models.eorm import (
    EORMModel,
    CoTEnergyInput,
    _make_token_sequence,
    _SEP_ID,
    _layer_norm,
    _transformer_layer_forward,
)
from carnot.models.ebm_cot_calibrator_v3 import SyntheticCoTPairGenerator


def get_layer_energies(model: EORMModel, cot_input: CoTEnergyInput) -> list[float]:
    """
    Extract the EORM energy score at each transformer layer by applying the 
    final layer norm and readout head to the intermediate hidden states.
    """
    token_ids = _make_token_sequence(
        cot_input.question_text,
        cot_input.response_text,
        model.max_seq_len,
        model.vocab_size,
    )
    if not token_ids:
        token_ids = [_SEP_ID]
    
    seq_len = len(token_ids)
    token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
    pos_ids = jnp.arange(seq_len, dtype=jnp.int32)

    x = model.params["token_embed"][token_ids_arr] + model.params["pos_embed"][pos_ids]
    
    energies = []
    
    def calc_energy(hidden_seq: jax.Array) -> float:
        x_norm = _layer_norm(hidden_seq, model.params["final_ln_gamma"], model.params["final_ln_beta"])
        pooled = jnp.mean(x_norm, axis=0)
        e = jnp.dot(pooled, model.params["out_weight"]) + model.params["out_bias"][0]
        return float(e)

    for lp in model.params["layers"]:
        x = _transformer_layer_forward(x, lp, model.n_heads)
        energies.append(calc_energy(x))
        
    return energies


def find_optimal_exit_layer(energies: list[float], threshold: float = 0.5) -> int:
    """
    Identify the earliest layer where the energy is within `threshold` 
    of the final layer's energy.
    """
    if not energies:
        return 0
    final_e = energies[-1]
    for i, e in enumerate(energies):
        if abs(e - final_e) < threshold:
            return i
    return len(energies) - 1


def run_experiment(model: EORMModel, out_path: str) -> None:
    """
    Run Experiment 1811: track Langevin energy gradients across model layers 
    to identify early-exit thresholds.
    """
    generator = SyntheticCoTPairGenerator(model, n_samples=10)
    pairs = generator.generate()
    
    exit_layers = []
    layer_energy_trajectories = []
    
    for text, is_correct in pairs:
        cot = CoTEnergyInput(question_text="Context", response_text=text)
        energies = get_layer_energies(model, cot)
        
        opt_layer = find_optimal_exit_layer(energies)
        exit_layers.append(opt_layer)
        layer_energy_trajectories.append(energies)
        
    distribution = Counter(exit_layers)
    
    # Format distribution keys as strings for JSON
    dist_dict = {str(k): v for k, v in distribution.items()}
    mean_layer = float(jnp.mean(jnp.array(exit_layers)))
    
    results = {
        "experiment": "1811",
        "name": "early_exit_langevin_energy_tracking",
        "optimal_exit_layer_distribution": dist_dict,
        "mean_optimal_layer": mean_layer,
        "layer_energy_trajectories": layer_energy_trajectories,
    }
    
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Run the experiment with a standard EORM model when executed directly
    model = EORMModel()
    out_file = "results/experiment_1811_early_exit.json"
    run_experiment(model, out_file)
    print(f"Experiment 1811 finished. Results written to {out_file}")

import time
import numpy as np
import json
from datetime import datetime

from carnot.samplers.backend import get_backend
from carnot.samplers.dtm import DtmBackend

def eval_energy(samples, b, J):
    return -0.5 * np.einsum('ni,ij,nj->n', samples, J, samples) - np.einsum('ni,i->n', samples, b)

def main():
    n_vars = 256
    rng = np.random.default_rng(42)
    biases = rng.normal(0, 1, size=n_vars)
    couplings = rng.normal(0, 1, size=(n_vars, n_vars))
    couplings = (couplings + couplings.T) / 2
    np.fill_diagonal(couplings, 0)
    
    # 1. Baseline Gibbs
    gibbs_backend = get_backend("cpu")
    t0 = time.time()
    gibbs_samples = gibbs_backend.minimize_energy(biases, couplings, n_samples=100, n_steps=1000, beta=10.0)
    gibbs_delay = time.time() - t0
    gibbs_energies = eval_energy(gibbs_samples, biases, couplings)
    gibbs_mean = float(np.mean(gibbs_energies))
    gibbs_min = float(np.min(gibbs_energies))
    
    # 2. DTM Backend
    dtm_backend = DtmBackend()
    t0 = time.time()
    dtm_samples = dtm_backend.minimize_energy(biases, couplings, n_samples=100, n_steps=1000, beta=10.0)
    dtm_delay = time.time() - t0
    dtm_energies = eval_energy(dtm_samples, biases, couplings)
    dtm_mean = float(np.mean(dtm_energies))
    dtm_min = float(np.min(dtm_energies))
    
    # Deficiency: difference from optimal/baseline
    deficiency = max(0.0, dtm_mean - gibbs_min)
    
    # EDDP (Energy-Delay-Deficiency) metric
    # energy_cost: absolute energy or energy drawn, let's use abs(dtm_mean)
    # let's define EDDP = |mean_energy| * delay * deficiency
    dtm_eddp = abs(dtm_mean) * dtm_delay * deficiency
    gibbs_eddp = abs(gibbs_mean) * gibbs_delay * max(0.0, gibbs_mean - gibbs_min)
    
    res = {
        "experiment": 1949,
        "schema": "carnot.experiment.v1",
        "title": "Denoising Thermodynamics",
        "run_date": "20260512",
        "started_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success",
        "baseline_gibbs": {
            "min_energy": gibbs_min,
            "mean_energy": gibbs_mean,
            "delay_s": gibbs_delay,
            "eddp": gibbs_eddp
        },
        "dtm": {
            "min_energy": dtm_min,
            "mean_energy": dtm_mean,
            "delay_s": dtm_delay,
            "eddp": dtm_eddp
        },
        "thrml_import_ready": True
    }
    
    with open("results/experiment_1949_denoising_thermodynamics.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()

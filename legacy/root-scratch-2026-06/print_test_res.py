import json
with open("results/experiment_3327_energy_descent_substrate_bootstrap_v1.json") as f:
    print(json.dumps(json.load(f)["blocked_reasons"]))

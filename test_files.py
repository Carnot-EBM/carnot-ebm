import json
with open("results/experiment_2395_fregelogic.json") as f:
    d = json.load(f)
    print("fregelogic:", "scores" in d, d.get("fregelogic_auroc"))
with open("results/experiment_2423_hierarchical_logcons_v2.json") as f:
    d = json.load(f)
    print("hierarchical:", "scores" in d, d.get("logcons_auroc"))

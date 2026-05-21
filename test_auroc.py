import json
import numpy as np
from run_semantic_energy_eval import main
import run_semantic_energy_eval
with open('results/experiment_2731_semantic_energy_tier0g.json') as f:
    d = json.load(f)
    print("AUROC:", d['tier0g_auroc'])

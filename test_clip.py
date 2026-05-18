import numpy as np
import scipy.stats
import sys
sys.path.append("python")
from carnot.verify.hive_ensemble import _binary_auroc
from carnot.verify.conformal_ensemble import build_experiment_artifact, ConformalEnsemble

art = build_experiment_artifact()
# Wait, build_experiment_artifact runs with whatever is in conformal_ensemble.py

from carnot.verify.conformal_ensemble import build_experiment_artifact, DEFAULT_MANIFEST_PATH, HiveEnsembleDetector, _read_jsonl, label_from_entry, _preconditions, robust_load_json
from pathlib import Path
import numpy as np

manifest_path = Path(DEFAULT_MANIFEST_PATH)
entries = _read_jsonl(manifest_path, limit=36)
labels = np.array([label_from_entry(e) for e in entries])
detector = HiveEnsembleDetector(random_seed=42)
raw_scores = detector.collect_verifier_scores(entries, labels.tolist())
extras = [
    ("experiment_2435_tier0k_scores.json", "tier0k_diffutruth"),
    ("experiment_2436_tier0l_scores.json", "tier0l_pcib"),
    ("experiment_2437_logcons_z3_scores.json", "logcons_z3")
]
for f_name, v_name in extras:
    p = Path(f"results/{f_name}")
    if p.exists():
        data = robust_load_json(p)
        scores = [x["score"] for x in sorted(data["scores"], key=lambda x: x["idx"])]
        raw_scores[v_name] = scores
print(raw_scores.keys())

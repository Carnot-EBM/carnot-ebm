import sys
from pathlib import Path
from carnot.verify.and_composition_verifier import build_default_verifier_ensemble
from carnot.eval.fover_memory_leakage_v3 import _read_fover_rows, _select_balanced_subset, _label_to_int, compute_auroc

repo_root = Path(".")
rows = _select_balanced_subset(
    _read_fover_rows(repo_root / "data" / "fover_corpus.jsonl"),
    seed=42,
    n_examples=1000,
)
labels = [_label_to_int(row["label"]) for row in rows]
v = build_default_verifier_ensemble()

scores = []
for row in rows:
    res = v.verify("", row.get("step_text", ""))
    scores.append(max(res.per_verifier_scores.values()))

print("Seed 42 AUROC:", compute_auroc(labels, scores))

import sys
from pathlib import Path
from carnot.verify.and_composition_verifier import build_default_verifier_ensemble
from carnot.eval.fover_memory_leakage_v3 import _read_fover_rows, _select_balanced_subset, _label_to_int, compute_auroc

repo_root = Path(".")
all_rows = _read_fover_rows(repo_root / "data" / "fover_corpus.jsonl")

v = build_default_verifier_ensemble()
seeds = [42, 137, 271, 314, 1729]
aurocs = []
for seed in seeds:
    rows = _select_balanced_subset(all_rows, seed=seed, n_examples=1000)
    labels = [_label_to_int(row["label"]) for row in rows]
    scores = []
    for row in rows:
        res = v.verify("", row.get("step_text", ""))
        scores.append(max(res.per_verifier_scores.values()))
    auroc = compute_auroc(labels, scores)
    print(f"Seed {seed} AUROC:", auroc)
    aurocs.append(auroc)
print("Mean:", sum(aurocs)/len(aurocs))

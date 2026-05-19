import json
import numpy as np
from pathlib import Path
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import sys

# Support running directly from the project root
try:
    from carnot.verify.nco_constraint import compute_nco_rejection_rate
except ImportError:
    sys.path.insert(0, 'python')
    from carnot.verify.nco_constraint import compute_nco_rejection_rate

def robust_load_json(p: Path) -> dict:
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _idx = decoder.raw_decode(text)
        return obj

def compute_p_values(X_cal: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    n_test, n_verifiers = X_test.shape
    n_cal = X_cal.shape[0]
    p_values = np.zeros((n_test, n_verifiers))
    for i in range(n_verifiers):
        cal_scores = np.sort(X_cal[:, i])
        for j in range(n_test):
            score_x = X_test[j, i]
            count = np.sum(cal_scores >= score_x)
            p_values[j, i] = count / (n_cal + 1)
    return p_values

def fisher_combine(p_values: np.ndarray, clip_val: float = 1e-10) -> np.ndarray:
    p_values = np.clip(p_values, clip_val, 1.0)
    chi2_stat = -2 * np.sum(np.log(p_values), axis=1)
    df = 2 * p_values.shape[1]
    p_combined = scipy.stats.chi2.sf(chi2_stat, df=df)
    return 1.0 - p_combined

def normalize_label(lbl: str) -> int:
    if lbl == 'correct': return 0
    if lbl == 'incorrect': return 1
    return int(lbl)

def get_scores(path: Path, key: str) -> list[float]:
    data = robust_load_json(path)
    if 'scores' in data:
        s = sorted(data['scores'], key=lambda x: x['idx'])
        return [float(x['score']) for x in s]
    if 'per_entry_results' in data:
        return [float(x[key]) for x in data['per_entry_results']]
    return []

def run_experiment(results_dir: Path = Path("results")) -> dict:
    preconditions_checked = ["sklearn_importable"]
    exp2484_file = results_dir / "experiment_2484_auroc_adversarial_replication.json"
    if exp2484_file.exists():
        preconditions_checked.append("exp2484_deliverable_present")
        exp2484_data = robust_load_json(exp2484_file)
        baseline_iso = exp2484_data["true_replicated_auroc_isotonic"]
    else:
        baseline_iso = 0.7964285714285715

    manifest = results_dir / 'live_sota_balanced_telemetry_manifest_1480.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().strip().split('\n')]
    y = np.array([normalize_label(r['correctness_label']) for r in rows], dtype=int)

    # Group A: SemanticEnergy, LaaB Meta-Judgment, FregeLogic
    A1 = get_scores(results_dir / 'experiment_2395_fregelogic.json', 'semantic_energy_score')
    A2 = get_scores(results_dir / 'experiment_2450_laab_meta_scores.json', 'score')
    A3 = get_scores(results_dir / 'experiment_2395_fregelogic.json', 'fregelogic_risk_score')
    X_A = np.column_stack([A1, A2, A3]).astype(np.float64)

    # Group B: DiffuTruth, PCIB, HalluField/HALT
    B1 = get_scores(results_dir / 'experiment_2435_tier0k_scores.json', 'score')
    B2 = get_scores(results_dir / 'experiment_2436_tier0l_scores.json', 'score')
    B3 = get_scores(results_dir / 'experiment_2449_tier0m_scores.json', 'score')
    X_B = np.column_stack([B1, B2, B3]).astype(np.float64)

    # Group C: LogCons Hierarchical, NCO, Tier0n Internal Repr
    C1 = get_scores(results_dir / 'experiment_2437_logcons_z3_scores.json', 'score')
    C2 = [compute_nco_rejection_rate(r.get('token_logprobs', [])) for r in rows]
    C3 = get_scores(results_dir / 'experiment_2460_tier0n_scores.json', 'score')
    X_C = np.column_stack([C1, C2, C3]).astype(np.float64)

    seeds = [42, 123, 456, 789, 1337]
    seed_results = []
    ensemble_aurocs = []

    for seed in seeds:
        idx = np.arange(len(y))
        idx_train, idx_test, y_train, y_test = train_test_split(
            idx, y, test_size=0.3, random_state=seed, stratify=y
        )

        calibrated_test_pvals = []
        group_aurocs_for_seed = []

        for X_group in [X_A, X_B, X_C]:
            X_train = X_group[idx_train]
            X_test = X_group[idx_test]

            p_train = compute_p_values(X_train, X_train)
            train_fisher = fisher_combine(p_train)

            p_test = compute_p_values(X_train, X_test)
            test_fisher = fisher_combine(p_test)

            isotonic_reg = IsotonicRegression(out_of_bounds='clip')
            isotonic_reg.fit(train_fisher, y_train)

            test_calibrated = isotonic_reg.predict(test_fisher)
            group_aurocs_for_seed.append(roc_auc_score(y_test, test_calibrated))

            calibrated_test_pvals.append(1.0 - test_calibrated)

        P_matrix = np.column_stack(calibrated_test_pvals)
        test_combined_fisher = fisher_combine(P_matrix)

        test_auroc = roc_auc_score(y_test, test_combined_fisher)
        ensemble_aurocs.append(test_auroc)

        seed_results.append({
            "seed": seed,
            "group_a_auroc": float(group_aurocs_for_seed[0]),
            "group_b_auroc": float(group_aurocs_for_seed[1]),
            "group_c_auroc": float(group_aurocs_for_seed[2]),
            "group_conditional_ensemble_auroc": float(test_auroc)
        })

    group_conditional_auroc_mean = float(np.mean(ensemble_aurocs))
    group_conditional_auroc_std = float(np.std(ensemble_aurocs))
    group_conditional_vs_isotonic_delta = float(group_conditional_auroc_mean - baseline_iso)
    group_conditional_vs_fisher_delta = float(group_conditional_auroc_mean - 0.9167)
    hive_peer_breached_group_cond = bool(group_conditional_auroc_mean > 0.9236)

    deliverable = {
        "group_conditional_auroc_mean": group_conditional_auroc_mean,
        "group_conditional_auroc_std": group_conditional_auroc_std,
        "group_conditional_vs_isotonic_delta": group_conditional_vs_isotonic_delta,
        "group_conditional_vs_fisher_delta": group_conditional_vs_fisher_delta,
        "hive_peer_breached_group_cond": hive_peer_breached_group_cond,
        "honest_verdict": f"complete: {group_conditional_auroc_mean:.4f}",
        "preconditions_checked": preconditions_checked,
        "results_by_seed": seed_results
    }
    
    out_path = results_dir / "experiment_2485_group_conformal_v5.json"
    out_path.write_text(json.dumps(deliverable, indent=2))
    print(f"Wrote {out_path}")
    return deliverable

if __name__ == "__main__":
    run_experiment()

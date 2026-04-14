"""Feature extraction from live verification traces for KAN training.

**Researcher summary:**
    Converts Exp 219-221 verification trace JSONs into binary feature vectors
    suitable for KANConstraintModel discriminative training.  The 13-bit
    encoding captures constraint-family satisfaction patterns, coverage,
    partial satisfaction, semantic violations, mode, output style, and model
    family — all the signals a verifier uses to judge correctness.

**Detailed explanation for engineers:**
    Exp 221 (constraint_ir) cases contain per-constraint satisfaction status
    across three families: literal (deterministic), search_optimization_limited
    (heuristic search), and semantic (LLM-judged).  We aggregate each family
    into a single binary "majority satisfied" feature.

    Additional features capture coverage (how many constraints were extracted),
    partial satisfaction (what fraction were individually met), whether any
    semantic violations occurred, the run mode (baseline/verify/repair), the
    output style (code vs prose), and which model family generated the response.

    The resulting 13-dim binary vector is the input to KANConstraintModel.
    Label: exact_satisfaction (1.0 = all constraints met, 0.0 = any failure).

FEATURE_DIM = 13 binary float32 features per case:
  [0]  literal_family_majority_sat    — majority of literal constraints satisfied
  [1]  search_opt_majority_sat        — majority of search_optimization_limited sat
  [2]  semantic_family_majority_sat   — majority of semantic constraints satisfied
  [3]  coverage_above_75              — extraction coverage > 0.75
  [4]  coverage_perfect               — extraction coverage == 1.0
  [5]  partial_above_50               — partial satisfaction > 0.50
  [6]  partial_above_75               — partial satisfaction > 0.75
  [7]  no_semantic_violations         — semantic_violation_count == 0
  [8]  mode_verify_only               — response was verify-only mode
  [9]  mode_verify_repair             — response was verify-repair mode
  [10] style_code_only                — output_style == 'code_only'
  [11] style_text_prose               — output_style == 'text_prose'
  [12] model_is_gemma                 — generating model is Gemma family

Label: exact_satisfaction (1.0 = satisfied, 0.0 = violated)

Spec: REQ-CORE-001, REQ-CORE-002
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Number of binary features per verification trace.
FEATURE_DIM: int = 13

# Constraint families recognised in Exp 221 constraint_ir traces.
_FAMILIES: tuple[str, ...] = (
    "literal",
    "search_optimization_limited",
    "semantic",
)


class TraceRecord(NamedTuple):
    """One extracted verification trace: feature vector and satisfaction label.

    Attributes:
        features: Binary float32 array of shape (FEATURE_DIM,).
        label: 1.0 if the response exactly satisfied all constraints, else 0.0.
    """

    features: np.ndarray  # shape (FEATURE_DIM,), dtype float32
    label: float          # 1.0 = satisfied, 0.0 = violated


def _majority(vals: list[bool]) -> float:
    """Return 1.0 if strictly more than half of vals are True, else 0.0.

    Returns 0.0 for an empty list (no evidence of satisfaction).

    Args:
        vals: Boolean satisfaction flags for one constraint family.

    Returns:
        1.0 if majority True, 0.0 otherwise.
    """
    if not vals:
        return 0.0
    return 1.0 if sum(vals) / len(vals) > 0.5 else 0.0


def extract_constraint_ir_features(
    case: dict,
    model_is_gemma: bool = False,
) -> np.ndarray:
    """Extract FEATURE_DIM-dimensional binary feature vector from a constraint_ir case.

    Reads the 'evaluation.constraint_results' list and aggregates per-family
    satisfaction into majority-vote binary features.  Additional features are
    derived from top-level case fields (coverage, partial_satisfaction, etc.).

    Args:
        case: One element from paired_run['cases'] in an Exp 221-style result JSON.
            Expected keys: 'evaluation', 'constraint_extraction_coverage',
            'partial_satisfaction', 'semantic_violation_count', 'mode',
            'output_style'.  Missing keys fall back to neutral defaults.
        model_is_gemma: True if the generating model is from the Gemma family.
            This is run-level information passed in by the caller.

    Returns:
        Binary float32 array of shape (FEATURE_DIM,).
    """
    eval_data = case.get("evaluation", {})
    constraint_results = eval_data.get("constraint_results", [])

    # Collect per-family satisfaction flags.
    family_sat: dict[str, list[bool]] = {f: [] for f in _FAMILIES}
    for cr in constraint_results:
        fam = cr.get("family", "")
        if fam in family_sat:
            family_sat[fam].append(cr.get("status", "") == "satisfied")

    coverage = float(case.get("constraint_extraction_coverage", 0.0))
    partial = float(case.get("partial_satisfaction", 0.0))
    violations = int(case.get("semantic_violation_count", 1))
    mode = str(case.get("mode", "baseline"))
    style = str(case.get("output_style", ""))

    return np.array(
        [
            _majority(family_sat["literal"]),
            _majority(family_sat["search_optimization_limited"]),
            _majority(family_sat["semantic"]),
            float(coverage > 0.75),
            float(coverage == 1.0),
            float(partial > 0.50),
            float(partial > 0.75),
            float(violations == 0),
            float(mode == "verify_only"),
            float(mode == "verify_repair"),
            float(style == "code_only"),
            float(style == "text_prose"),
            float(model_is_gemma),
        ],
        dtype=np.float32,
    )


def load_constraint_ir_traces(result_path: str | Path) -> list[TraceRecord]:
    """Load all verification traces from an Exp 221-style constraint_ir result JSON.

    Each paired_run contributes model+mode context; each case in the run yields
    one TraceRecord.  Model family (Gemma vs Qwen) is inferred from the run's
    'model_name' field.

    Args:
        result_path: Path to the experiment result JSON
            (e.g. 'results/experiment_221_results.json').

    Returns:
        List of TraceRecord — one per (run, case) pair.  Order is stable:
        all cases in paired_runs[0], then paired_runs[1], etc.
    """
    with open(result_path) as fh:
        data = json.load(fh)

    records: list[TraceRecord] = []
    for run in data.get("paired_runs", []):
        model_name = run.get("model_name", "")
        model_is_gemma = "gemma" in model_name.lower()
        for case in run.get("cases", []):
            features = extract_constraint_ir_features(case, model_is_gemma=model_is_gemma)
            label = float(bool(case.get("exact_satisfaction", False)))
            records.append(TraceRecord(features=features, label=label))

    return records


def auroc_score(
    energies_correct: np.ndarray,
    energies_wrong: np.ndarray,
) -> float:
    """Compute AUROC via Wilcoxon-Mann-Whitney U statistic.

    Convention: **lower energy = more correct**.  AUROC measures
    P(E_correct < E_wrong) over all (correct, wrong) pairs.

    Args:
        energies_correct: Energy values for exactly-satisfied (correct) cases.
        energies_wrong: Energy values for constraint-violated (wrong) cases.

    Returns:
        AUROC in [0, 1].  0.5 = random chance, 1.0 = perfect discrimination.
        Returns 0.5 when either array is empty (undefined).
    """
    n_c = len(energies_correct)
    n_w = len(energies_wrong)
    if n_c == 0 or n_w == 0:
        return 0.5

    wins = 0.0
    pairs = n_c * n_w
    for ec in energies_correct:
        for ew in energies_wrong:
            if ec < ew:
                wins += 1.0
            elif ec == ew:
                wins += 0.5

    return wins / pairs

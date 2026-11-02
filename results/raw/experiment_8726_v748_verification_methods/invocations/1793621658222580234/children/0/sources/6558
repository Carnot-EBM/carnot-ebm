#!/usr/bin/env python3
"""Verifier-as-DETECTOR AUROC probe (operator-chosen direction, 2026-06-14).

Isolates the verifier's DETECTION value (can it flag/reject wrong outputs?) from its
SELECTION value (oracle@K-vote, headroom-conditional). On Sudoku the executable verifier
(constraint satisfaction) should separate correct from incorrect outputs with high AUROC
EVEN at the converged checkpoint where selection headroom was ~0 (exp v3 headroom curve) --
the headline contrast: detection works where selection can't.

HONEST framing (Circularity Discipline): on Sudoku the verifier IS the executable oracle
(constraint check ~ correctness for a unique-solution puzzle), so this is an
EXECUTION_GROUNDED / verifier_is_oracle=TRUE detection result -- valid, but not a
headline moat. The point is the detection-vs-selection DIVERGENCE, plus the abstention
curve (accuracy vs coverage), which is the actually-useful capability (precision / "I don't
know"). The non-circular detector (diffusion-surprisal / a learned detector) is future work.

Output: results/verifier_detector_auroc.json
"""

from __future__ import annotations

import json
import gzip
import hashlib
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from carnot.paths import repo_root

_orig_load = torch.load
torch.load = lambda *a, **k: k.update(weights_only=False) or _orig_load(*a, **k)  # trusted ckpt

NANO_TRM = "/home/ianblenke/github.com/ianblenke/carnot/nano-trm"
sys.path.insert(0, NANO_TRM)
from src.nn.sudoku_evaluator import SudokuEvaluator  # noqa: E402
from src.nn.utils.constants import IGNORE_LABEL_ID  # noqa: E402

# Sudoku-Extreme token encoding: token = digit + 2 (labels are {3..11} for digits 1..9;
# input token 2 = blank). Decode digit = token - TOKEN_OFFSET before constraint checks.
TOKEN_OFFSET = 2

STABLE = (
    "/home/ianblenke/github.com/ianblenke/carnot/results/trm_runs/sudoku_extreme_baseline/last.ckpt"
)
DATA_DIR = f"{NANO_TRM}/data/sudoku_extreme_1k_aug_1k"
# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
ROOT = repo_root()
OUT = ROOT / "results/experiment_4208_verifier_as_detector_auroc.json"
REQUIRED_CACHED_POOLS = [
    ROOT / "results/arc3_trm_verifier_rerank.json",
    ROOT / "results/experiment_4175_headroom_gate_executable_census.json",
]
CODE_POOL = ROOT / "results/experiment_1999_code_verification_humaneval.json"
MATH_POOL = ROOT / "data/p01_difficulty_matched_generations.jsonl"
ARC_POOL = ROOT / "results/arc3_gap3_stage2_eval_pool.json.gz"
ARC_PROGRAMS = ROOT / "results/arc3_gap4_induced_programs.json"
SUDOKU_HEADROOM = ROOT / "results/trm_runs/v3_headroom_probe_0.82.json"
RANDOM_SEED = 4208
BOOTSTRAP_N = 1000


def constraint_sat_fraction(grid: torch.Tensor, n: int = 9) -> float:
    """Fraction of row/col/box all-distinct-1..n constraints satisfied. grid: [n,n] ints."""
    box = 3
    total = 3 * n  # n rows + n cols + n boxes
    ok = 0
    for r in range(n):
        row = grid[r]
        if torch.all((row >= 1) & (row <= n)) and len(torch.unique(row)) == n:
            ok += 1
    for c in range(n):
        col = grid[:, c]
        if torch.all((col >= 1) & (col <= n)) and len(torch.unique(col)) == n:
            ok += 1
    for br in range(0, n, box):
        for bc in range(0, n, box):
            b = grid[br : br + box, bc : bc + box].reshape(-1)
            if torch.all((b >= 1) & (b <= n)) and len(torch.unique(b)) == n:
                ok += 1
    return ok / total


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney). labels: 1=positive(correct)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), ndigits)


def unavailable_report(reason: str) -> dict[str, Any]:
    return {
        "n": 0,
        "auroc": None,
        "ci95": [None, None],
        "base_rate": None,
        "brier": None,
        "precision_at_recall_0.9": {
            "precision": None,
            "recall": None,
            "coverage": None,
            "threshold": None,
        },
        "abstention_curve": [],
        "random_auroc": None,
        "random_auroc_ci95": [None, None],
        "unavailable_reason": reason,
    }


def bootstrap_ci95(
    scores: list[float],
    labels: list[int],
    *,
    seed: int = RANDOM_SEED,
    bootstrap_n: int = BOOTSTRAP_N,
) -> list[float | None]:
    if len(set(labels)) < 2 or not scores:
        return [None, None]
    rng = random.Random(seed)
    n = len(scores)
    vals: list[float] = []
    for _ in range(bootstrap_n):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_scores = [scores[i] for i in idx]
        sample_labels = [labels[i] for i in idx]
        val = auroc(sample_scores, sample_labels)
        if not math.isnan(val):
            vals.append(float(val))
    if not vals:
        return [None, None]
    vals.sort()
    lo_i = int(0.025 * (len(vals) - 1))
    hi_i = int(0.975 * (len(vals) - 1))
    return [_round_or_none(vals[lo_i]), _round_or_none(vals[hi_i])]


def brier_score(scores: list[float], labels: list[int]) -> float | None:
    if not scores:
        return None
    return sum((min(1.0, max(0.0, s)) - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def precision_at_recall(
    scores: list[float],
    labels: list[int],
    *,
    target_recall: float = 0.9,
) -> dict[str, float | None]:
    total_pos = sum(labels)
    if total_pos == 0 or not scores:
        return {"precision": None, "recall": None, "coverage": None, "threshold": None}
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    seen_pos = 0
    for keep, idx in enumerate(order, start=1):
        seen_pos += labels[idx]
        recall = seen_pos / total_pos
        if recall >= target_recall:
            return {
                "precision": _round_or_none(seen_pos / keep),
                "recall": _round_or_none(recall),
                "coverage": _round_or_none(keep / len(scores)),
                "threshold": _round_or_none(scores[idx]),
            }
    return {"precision": None, "recall": None, "coverage": None, "threshold": None}


def abstention_curve(
    scores: list[float],
    labels: list[int],
    coverages: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25),
) -> list[dict[str, float | int | None]]:
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    curve = []
    for coverage in coverages:
        keep_n = max(1, int(len(order) * coverage))
        keep = order[:keep_n]
        curve.append(
            {
                "coverage": coverage,
                "n_kept": len(keep),
                "accuracy": _round_or_none(sum(labels[i] for i in keep) / len(keep)),
            }
        )
    return curve


def score_rows_to_report(
    scores: list[float],
    labels: list[int],
    *,
    seed: int = RANDOM_SEED,
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    if not scores or len(set(labels)) < 2:
        return unavailable_report("need_at_least_one_correct_and_one_wrong")
    rng = random.Random(seed)
    random_scores = [rng.random() for _ in scores]
    det_auroc = auroc(scores, labels)
    rnd_auroc = auroc(random_scores, labels)
    return {
        "n": len(scores),
        "auroc": _round_or_none(det_auroc),
        "ci95": bootstrap_ci95(scores, labels, seed=seed, bootstrap_n=bootstrap_n),
        "base_rate": _round_or_none(sum(labels) / len(labels)),
        "brier": _round_or_none(brier_score(scores, labels)),
        "precision_at_recall_0.9": precision_at_recall(scores, labels),
        "abstention_curve": abstention_curve(scores, labels),
        "random_auroc": _round_or_none(rnd_auroc),
        "random_auroc_ci95": bootstrap_ci95(
            random_scores,
            labels,
            seed=seed + 1,
            bootstrap_n=bootstrap_n,
        ),
    }


def valid_but_wrong_report(
    scores: list[float],
    labels: list[int],
    valid_flags: list[bool],
    *,
    seed: int = RANDOM_SEED,
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    idx = [i for i, valid in enumerate(valid_flags) if valid]
    subset_scores = [scores[i] for i in idx]
    subset_labels = [labels[i] for i in idx]
    if len(set(subset_labels)) < 2:
        return {
            "valid_but_wrong_n": len(idx),
            "valid_but_wrong_negatives": sum(1 for y in subset_labels if y == 0),
            "valid_but_wrong_auroc": None,
            "valid_but_wrong_auroc_ci95": [None, None],
            "valid_but_wrong_note": "need both valid-correct and valid-wrong outputs",
        }
    return {
        "valid_but_wrong_n": len(idx),
        "valid_but_wrong_negatives": sum(1 for y in subset_labels if y == 0),
        "valid_but_wrong_auroc": _round_or_none(auroc(subset_scores, subset_labels)),
        "valid_but_wrong_auroc_ci95": bootstrap_ci95(
            subset_scores,
            subset_labels,
            seed=seed,
            bootstrap_n=bootstrap_n,
        ),
    }


def _scores_and_labels(rows: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    return [float(r["score"]) for r in rows], [int(r["is_correct"]) for r in rows]


def load_code_rows(path: Path = CODE_POOL) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = []
    for item in data.get("results", []):
        task = item.get("task_id", f"task{len(rows)}")
        for variant, key in (("baseline", "baseline_passed"), ("repair", "repair_passed")):
            if key not in item:
                continue
            passed = bool(item[key])
            rows.append(
                {
                    "domain": "code",
                    "output": f"{task}:{variant}",
                    "score": 1.0 if passed else 0.0,
                    "is_correct": 1 if passed else 0,
                    "score_kind": "unit_test_pass_flag",
                }
            )
    return rows


def load_math_rows(path: Path = MATH_POOL) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            problem = item.get("problem_id", f"math:{line_no}")
            candidates = []
            if isinstance(item.get("greedy"), dict):
                candidates.append(("greedy", item["greedy"]))
            candidates.extend((f"sample{i}", c) for i, c in enumerate(item.get("samples", [])))
            for variant, cand in candidates:
                if "correct" not in cand:
                    continue
                correct = bool(cand["correct"])
                rows.append(
                    {
                        "domain": "math",
                        "output": f"{problem}:{variant}",
                        "score": 1.0 if correct else 0.0,
                        "is_correct": 1 if correct else 0,
                        "score_kind": "exact_answer_match",
                    }
                )
    return rows


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _norm_hamming(cand: list[list[int]], pred: list[list[int]]) -> float:
    ch, cw = _grid_shape(cand)
    ph, pw = _grid_shape(pred)
    c_size = max(1, ch * cw)
    p_size = max(1, ph * pw)
    if (ch, cw) != (ph, pw):
        return 2.0
    diff = 0
    for r in range(ch):
        for c in range(cw):
            diff += int(cand[r][c] != pred[r][c])
    return diff / max(1, c_size)


def arc_rows_from_entries(
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_entry = {int(p.get("entry_i", i)): p for i, p in enumerate(programs) if isinstance(p, dict)}
    rows = []
    for entry_i, entry in enumerate(entries):
        prog = by_entry.get(entry_i, {})
        pred = prog.get("pred_grid")
        fit = float(prog.get("demo_fit") or 0.0)
        for cand_i, cand in enumerate(entry.get("candidates", [])):
            if pred is None:
                score = 0.0
            else:
                hamming = _norm_hamming(cand["grid"], pred)
                score = fit * max(0.0, 1.0 - min(hamming, 2.0) / 2.0)
            rows.append(
                {
                    "domain": "arc",
                    "output": f"{entry.get('task', entry_i)}:candidate{cand_i}",
                    "score": float(score),
                    "is_correct": 1 if cand.get("correct") else 0,
                    "score_kind": "gap4_demo_fit_execution_consistency",
                }
            )
    return rows


def load_arc_rows(
    pool_path: Path = ARC_POOL,
    programs_path: Path = ARC_PROGRAMS,
) -> list[dict[str, Any]]:
    with gzip.open(pool_path, "rt") as f:
        pool = json.load(f)
    programs = json.loads(programs_path.read_text()).get("programs", [])
    return arc_rows_from_entries(pool.get("entries", []), programs)


def hash_source_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted({Path(p) for p in paths}, key=lambda p: str(p)):
        h.update(str(path).encode())
        if not path.exists():
            h.update(b"\0MISSING\0")
            continue
        if path.is_dir():
            h.update(b"\0DIR\0")
            continue
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_selector_headroom() -> dict[str, float | None]:
    census = json.loads(REQUIRED_CACHED_POOLS[1].read_text())
    arc = json.loads(REQUIRED_CACHED_POOLS[0].read_text())
    sudoku_headroom = None
    if SUDOKU_HEADROOM.exists():
        sudoku_headroom = json.loads(SUDOKU_HEADROOM.read_text()).get("best_headroom")
    per = census.get("per_domain_headroom", {})
    return {
        "sudoku": _round_or_none(sudoku_headroom),
        "code": _round_or_none(per.get("code", {}).get("selectable_headroom")),
        "math": _round_or_none(per.get("math", {}).get("selectable_headroom")),
        "arc": _round_or_none(arc.get("present_but_misvoted_headroom")),
    }


def _detection_beats_random(report: dict[str, Any]) -> bool:
    ci = report.get("ci95") or [None, None]
    return bool(ci[0] is not None and ci[0] > 0.5)


def build_artifact(
    *,
    domain_reports: dict[str, dict[str, Any]],
    selector_headroom: dict[str, float | None],
    verifier_is_oracle: dict[str, bool],
    decode_sanity: dict[str, Any],
    source_paths: list[Path],
    duration_s: float,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    domains = ["sudoku", "code", "math", "arc"]
    detection = {d: domain_reports.get(d, {}).get("auroc") for d in domains}
    ci95 = {d: domain_reports.get(d, {}).get("ci95", [None, None]) for d in domains}
    controls = {}
    for d in domains:
        rep = domain_reports.get(d, unavailable_report("missing_report"))
        controls[d] = {
            "base_rate": rep.get("base_rate"),
            "random_auroc": rep.get("random_auroc"),
            "random_auroc_ci95": rep.get("random_auroc_ci95"),
        }
        if d == "sudoku":
            controls[d]["valid_but_wrong_auroc"] = rep.get("valid_but_wrong_auroc")
            controls[d]["valid_but_wrong_auroc_ci95"] = rep.get("valid_but_wrong_auroc_ci95")
            controls[d]["valid_but_wrong_n"] = rep.get("valid_but_wrong_n")
            controls[d]["valid_but_wrong_negatives"] = rep.get("valid_but_wrong_negatives")
            controls[d]["valid_but_wrong_note"] = rep.get("valid_but_wrong_note")

    divergence_domains = [
        d
        for d in domains
        if detection.get(d) is not None
        and selector_headroom.get(d) is not None
        and selector_headroom[d] <= 0.01
        and _detection_beats_random(domain_reports[d])
    ]
    if divergence_domains:
        verdict = "complete: detector_selection_divergence_" + "_".join(divergence_domains)
    elif any(v is not None for v in detection.values()):
        verdict = "complete: detector_axis_measured_no_ci_exclusive_zero_headroom_divergence"
    else:
        verdict = "complete: detector_axis_no_scored_rows_available"

    return {
        "experiment": "experiment_4208_verifier_as_detector_auroc",
        "schema": "carnot.verifier_detector_auroc.v1",
        "honest_verdict": verdict,
        "field_principles": {
            "honest_verdict": "Terminal-prefixed. A clean per-domain detector measurement (high AUROC where selection had none, OR AUROC~0.5 everywhere) is a COMPLETE, decision-grade result either way.",
            "detection_auroc_by_domain": "BARE per-domain {domain: auroc} -- the detection axis the selector metric cannot see; value for abstention/precision even at zero selection headroom.",
            "selector_headroom_by_domain": "BARE per-domain {domain: oracle@K - vote} from .387/.388 -- the contrast axis; the headline is detection >> 0.5 WHERE selector_headroom ~0.",
            "verifier_is_oracle_by_domain": "Per-domain bool -- on code/Sudoku the executable check IS the oracle (circular/execution-grounded detector); declares honestly which domains' detection is oracle-distinct (Circularity Discipline).",
            "controls": "{base_rate, random_auroc, valid_but_wrong_auroc(Sudoku)} -- the positive/negative controls + the degenerate-detector guard that rule out a trivially-high AUROC (FALSE_NEGATIVE_RISK / the invalid-grid trap).",
            "random_seed": "Determinism precondition; the bootstrap AUROC must be reproducible.",
            "reproducibility_checksum": "Hash of the cached pools scored; catches silent pool drift.",
        },
        "inference_substrate": "cached_pool_scoring_plus_live_trm_checkpoint_decode_no_llm_generation",
        "spec_refs": ["REQ-VERIFY-4208", "SCENARIO-VERIFY-4208"],
        "detection_auroc_by_domain": detection,
        "detection_auroc_ci95_by_domain": ci95,
        "selector_headroom_by_domain": selector_headroom,
        "n_by_domain": {d: domain_reports.get(d, {}).get("n", 0) for d in domains},
        "verifier_is_oracle_by_domain": verifier_is_oracle,
        "metrics_by_domain": domain_reports,
        "controls": controls,
        "detection_beats_random_ci95_exclusive_by_domain": {
            d: _detection_beats_random(domain_reports.get(d, {})) for d in domains
        },
        "divergence_domains": divergence_domains,
        "decode_sanity": decode_sanity,
        "random_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_N,
        "reproducibility_checksum": hash_source_paths(source_paths),
        "source_paths": [str(p) for p in source_paths],
        "duration_s": _round_or_none(duration_s, 3),
    }


def blocked_artifact(reason: str, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4208_verifier_as_detector_auroc",
        "honest_verdict": reason,
        "spec_refs": ["REQ-VERIFY-4208", "SCENARIO-VERIFY-4208"],
        "preconditions_checked": [
            {"path": str(p), "available": p.exists()} for p in REQUIRED_CACHED_POOLS
        ],
        "duration_s": _round_or_none(duration_s, 3),
        "random_seed": RANDOM_SEED,
    }


@torch.no_grad()
def run_sudoku_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover
    t0 = time.time()
    ev = SudokuEvaluator(
        checkpoint_path=STABLE, data_dir=DATA_DIR, batch_size=128, device="auto", eval_split="test"
    )
    ev.datamodule.setup("test")
    loader = ev.datamodule.test_dataloader()
    m = ev.model
    m.eval()
    n = 9

    # --- decoding sanity check: gold labels must be valid Sudoku (confirms token==digit) ---
    sanity = {"gold_valid_frac": None}
    first = next(iter(loader))
    lab0 = first["output"][:16]
    gv = 0
    for i in range(lab0.shape[0]):
        g = lab0[i].reshape(n, n).long() - TOKEN_OFFSET
        if constraint_sat_fraction(g, n) == 1.0:
            gv += 1
    sanity["gold_valid_frac"] = gv / lab0.shape[0]

    rows = []
    scores, labels, valid_flags = [], [], []
    greedy_correct = 0
    total = 0
    for batch in loader:
        b = {k: v.to(ev.device) for k, v in batch.items()}
        carry = m.initial_carry(b)
        steps = 0
        logits = None
        while True:
            carry, out = m.forward(carry, b)
            logits = out["logits"]
            steps += 1
            if carry.halted.all() or steps > 64:
                break
        pred = logits.argmax(-1)  # [B, 81]
        label = carry.current_data["output"]
        mask = label != IGNORE_LABEL_ID
        B = label.shape[0]
        for i in range(B):
            p = pred[i]
            lab = label[i]
            mk = mask[i]
            is_corr = bool(((p == lab) | (~mk)).all().item())
            grid = p.reshape(n, n).long() - TOKEN_OFFSET  # decode tokens->digits (token=digit+2)
            vscore = constraint_sat_fraction(grid, n)
            exact_valid = bool(ev.check_sudoku_validity(grid, n))
            scores.append(vscore)
            labels.append(1 if is_corr else 0)
            valid_flags.append(exact_valid)
            rows.append(
                {
                    "domain": "sudoku",
                    "output": f"test:{total}",
                    "score": float(vscore),
                    "is_correct": 1 if is_corr else 0,
                    "valid": exact_valid,
                    "score_kind": "satisfied_constraints_fraction",
                }
            )
            greedy_correct += int(is_corr)
            total += 1

    greedy_acc = greedy_correct / max(1, total)
    sanity.update(
        {
            "greedy_exact_accuracy": round(greedy_acc, 4),
            "expected_test_accuracy_band": [0.74, 0.85],
            "checked": bool(0.74 <= greedy_acc <= 0.85),
            "checkpoint": STABLE,
            "duration_s": round(time.time() - t0, 1),
        }
    )
    return rows, sanity


def main() -> None:  # pragma: no cover
    started = time.time()
    if not all(p.exists() for p in REQUIRED_CACHED_POOLS):
        art = blocked_artifact("blocked_detector_cached_pools_missing", time.time() - started)
        OUT.write_text(json.dumps(art, indent=2) + "\n")
        print(f"[detector] {art['honest_verdict']} -> {OUT}", flush=True)
        return

    domain_reports: dict[str, dict[str, Any]] = {}
    decode_sanity: dict[str, Any] = {}
    source_paths = [
        *REQUIRED_CACHED_POOLS,
        CODE_POOL,
        MATH_POOL,
        ARC_POOL,
        ARC_PROGRAMS,
        SUDOKU_HEADROOM,
        Path(STABLE),
    ]

    sudoku_rows, sudoku_sanity = run_sudoku_rows()
    decode_sanity["sudoku"] = sudoku_sanity
    scores, labels = _scores_and_labels(sudoku_rows)
    sudoku_report = score_rows_to_report(scores, labels, seed=RANDOM_SEED)
    sudoku_report.update(
        valid_but_wrong_report(
            scores,
            labels,
            [bool(r["valid"]) for r in sudoku_rows],
            seed=RANDOM_SEED + 100,
        )
    )
    sudoku_report["decode_sanity_checked"] = sudoku_sanity["checked"]
    domain_reports["sudoku"] = sudoku_report

    for domain, loader in (
        ("code", lambda: load_code_rows(CODE_POOL)),
        ("math", lambda: load_math_rows(MATH_POOL)),
        ("arc", lambda: load_arc_rows(ARC_POOL, ARC_PROGRAMS)),
    ):
        try:
            rows = loader()
        except FileNotFoundError as exc:
            domain_reports[domain] = unavailable_report(f"missing_rows:{exc.filename}")
            continue
        scores, labels = _scores_and_labels(rows)
        domain_reports[domain] = score_rows_to_report(
            scores,
            labels,
            seed=RANDOM_SEED + len(domain),
        )

    art = build_artifact(
        domain_reports=domain_reports,
        selector_headroom=load_selector_headroom(),
        verifier_is_oracle={"sudoku": True, "code": True, "math": True, "arc": False},
        decode_sanity=decode_sanity,
        source_paths=source_paths,
        duration_s=time.time() - started,
        seed=RANDOM_SEED,
    )
    OUT.write_text(json.dumps(art, indent=2) + "\n")
    print(
        "[detector] DONE "
        f"auroc={art['detection_auroc_by_domain']} "
        f"headroom={art['selector_headroom_by_domain']} -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

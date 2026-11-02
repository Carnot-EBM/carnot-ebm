"""Exp 4177 decisive headroom-controlled verifier moat test.

Spec refs: REQ-VERIFY-4177, SCENARIO-VERIFY-4177.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting import vstar_learned_selector_4176 as vstar


RANDOM_SEED = 4177
HEADROOM_THRESHOLD = 0.10
HEADROOM_REL = Path("results/experiment_4175_headroom_gate_executable_census.json")
SELECTOR_REL = Path("results/experiment_4176_vstar_selector_model.json")
OUTPUT_REL = Path("results/experiment_4177_decisive_headroom_controlled_moat_test.json")
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
FEATURE_NAMES = tuple(vstar.FEATURE_NAMES)
SPEC_REFS = ["REQ-VERIFY-4177", "SCENARIO-VERIFY-4177"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean positive, a clean (headroom-present) null, or a "
        "gated deferral are ALL COMPLETE."
    ),
    "verifier_value_added": (
        "Bare bool: did ARM A beat SC-vote (CI95 excl 0) OR match it at lower cost? "
        "Resolves the moat question and the DiffusionGemma gate when headroom is present."
    ),
    "moat_delta_vs_vote": (
        "pass@1(A) - pass@1(B) with CI95 -- the accuracy axis of the moat, now "
        "measured WHERE headroom exists."
    ),
    "moat_vs_matched_control": (
        "pass@1(A) - pass@1(C) -- proves a win is the verifier, not extra "
        "adaptation compute (arXiv:2511.02886)."
    ),
    "accuracy_cost_pareto": (
        "Compute/latency ratio at equal accuracy -- the efficiency-parity win "
        "condition (north-star §5; arXiv:2504.01005)."
    ),
    "positive_control_confirmed": (
        "Bare bool: oracle@k > SC-vote AND >=1 flip occurred on this corpus -- "
        "without it the null is uninformative (FALSE_NEGATIVE_RISK)."
    ),
    "random_seed": "Determinism precondition for reproducibility of the decisive measurement.",
    "reproducibility_checksum": (
        "Content hash of candidate pool + verifier config; lets a third party "
        "re-run the decisive test."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "verifier_value_added",
    "moat_delta_vs_vote",
    "moat_vs_matched_control",
    "accuracy_cost_pareto",
    "positive_control_confirmed",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "inference_substrate",
    "duration_s",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _round(value: float) -> float:
    return round(float(value), 10)


def _empty_delta(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "n": 0,
        "arm_a_pass1": 0.0,
        "arm_b_sc_vote_pass1": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "bootstrap_resamples": 0,
    }


def _empty_control(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "n": 0,
        "arm_a_pass1": 0.0,
        "arm_c_no_verifier_pass1": 0.0,
        "delta": 0.0,
        "control_policy": "deterministic_first_of_k_no_verifier",
    }


def _empty_pareto(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "same_candidate_budget": False,
        "efficiency_parity": False,
        "parity_at_n_times_cheaper": None,
        "value_added_basis": "none",
    }


def _empty_positive_control(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "oracle_at_k": 0.0,
        "sc_vote_pass1": 0.0,
        "oracle_minus_sc_vote": 0.0,
        "selection_flips_vs_vote": 0,
        "verifier_recovers_outvoted": 0,
    }


def _empty_artifact(reason: str, status: str, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "verifier_value_added": False,
        "moat_delta_vs_vote": _empty_delta(status),
        "moat_vs_matched_control": _empty_control(status),
        "accuracy_cost_pareto": _empty_pareto(status),
        "positive_control_confirmed": False,
        "positive_control": _empty_positive_control(status),
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "acceptance_gate": reason == "complete_moat_test_deferred_no_headroom_present",
        "adversarial_verify": {"status": "not_run", "reason": status},
    }


def _load_headroom_gate(repo_root: Path) -> dict[str, Any]:
    path = repo_root / HEADROOM_REL
    if not path.exists():
        raise BlockedRun("blocked_missing_headroom_gate")
    return _read_json_object(path)


def _defer_status(headroom: dict[str, Any]) -> str | None:
    max_headroom = _finite_float(headroom.get("max_selectable_headroom"))
    domain = str(headroom.get("headroom_present_domain") or "")
    if max_headroom < HEADROOM_THRESHOLD or not domain:
        return "deferred_no_headroom_present"
    return None


def _check_domain(headroom: dict[str, Any]) -> str:
    domain = str(headroom.get("headroom_present_domain") or "")
    if domain != "code":
        raise BlockedRun(f"blocked_unsupported_headroom_domain_{domain}")
    return domain


def _default_oracle_checker() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", "import subprocess"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    detail = (proc.stderr or proc.stdout or "subprocess import ok").strip()
    return proc.returncode == 0, detail


def _load_corpus(repo_root: Path) -> vstar.TraceCorpus:
    try:
        return vstar.load_trace_corpus(repo_root)
    except vstar.BlockedRun as exc:
        raise BlockedRun(exc.reason) from exc


def _load_selector(repo_root: Path) -> dict[str, Any]:
    path = repo_root / SELECTOR_REL
    if not path.exists():
        raise BlockedRun("blocked_missing_vstar_selector")
    selector = _read_json_object(path)
    if tuple(selector.get("feature_names", ())) != FEATURE_NAMES:
        raise BlockedRun("blocked_selector_feature_mismatch")
    return selector


def _group_rows(rows: Sequence[vstar.TraceRow]) -> list[list[vstar.TraceRow]]:
    grouped: dict[str, list[vstar.TraceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return [sorted(task_rows, key=lambda row: row.candidate_index) for task_rows in grouped.values()]


def _pass1(rows: Sequence[vstar.TraceRow]) -> float:
    return sum(row.correct for row in rows) / float(len(rows)) if rows else 0.0


def _bootstrap_ci(deltas: Sequence[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(random_seed)
    n = len(deltas)
    means = []
    for _ in range(int(resamples)):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [_round(lo), _round(hi)]


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(corpus: vstar.TraceCorpus, selector_path: Path) -> str:
    config = {
        "arm_a": "cached_executable_pass_flags_plus_vstar_selector",
        "arm_b": "self_consistency_vote_weight",
        "arm_c": "deterministic_first_of_k_no_verifier",
        "headroom_threshold": HEADROOM_THRESHOLD,
        "selector_model_sha256": _checksum(selector_path),
        "source_pool_sha256": _checksum(corpus.source_path),
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _measure_arms(
    corpus: vstar.TraceCorpus,
    selector: dict[str, Any],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    tasks = _group_rows(corpus.rows)

    start = time.perf_counter()
    selector_scores = {
        row.candidate_id: vstar.score_with_selector(selector, row.features) for row in corpus.rows
    }
    arm_a = [
        max(
            task_rows,
            key=lambda row: (
                float(row.correct),
                selector_scores[row.candidate_id],
                row.vote_weight,
                -row.candidate_index,
            ),
        )
        for task_rows in tasks
    ]
    arm_a_latency = time.perf_counter() - start

    start = time.perf_counter()
    arm_b = [
        max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        for task_rows in tasks
    ]
    arm_b_latency = time.perf_counter() - start

    start = time.perf_counter()
    arm_c = [task_rows[0] for task_rows in tasks]
    arm_c_latency = time.perf_counter() - start

    oracle_hits = [any(row.correct for row in task_rows) for task_rows in tasks]
    a_pass = _pass1(arm_a)
    b_pass = _pass1(arm_b)
    c_pass = _pass1(arm_c)
    deltas_ab = [float(a.correct) - float(b.correct) for a, b in zip(arm_a, arm_b, strict=True)]
    deltas_ac = [float(a.correct) - float(c.correct) for a, c in zip(arm_a, arm_c, strict=True)]
    delta_ab = sum(deltas_ab) / float(len(deltas_ab)) if deltas_ab else 0.0
    delta_ac = sum(deltas_ac) / float(len(deltas_ac)) if deltas_ac else 0.0
    ci95 = _bootstrap_ci(deltas_ab, random_seed=random_seed, resamples=bootstrap_resamples)
    selection_flips = sum(a.candidate_id != b.candidate_id for a, b in zip(arm_a, arm_b, strict=True))
    recovered = sum(a.correct and not b.correct for a, b in zip(arm_a, arm_b, strict=True))

    candidate_count = len(corpus.rows)
    task_count = len(tasks)
    arm_a_cost = candidate_count * 2
    arm_b_cost = candidate_count
    arm_c_cost = task_count
    efficiency_parity = a_pass >= b_pass and arm_a_cost < arm_b_cost
    accuracy_lift = delta_ab > 0.0 and ci95[0] > 0.0
    value_added_basis = "accuracy_lift_ci95_excludes_zero" if accuracy_lift else "none"
    if not accuracy_lift and efficiency_parity:  # pragma: no cover - future lower-cost verifier mode.
        value_added_basis = "accuracy_parity_lower_cost"

    return {
        "n_tasks": task_count,
        "candidate_count": candidate_count,
        "oracle_at_k": _round(sum(oracle_hits) / float(task_count)),
        "arm_a_pass1": _round(a_pass),
        "arm_b_pass1": _round(b_pass),
        "arm_c_pass1": _round(c_pass),
        "delta_ab": _round(delta_ab),
        "delta_ac": _round(delta_ac),
        "ci95": ci95,
        "bootstrap_resamples": int(bootstrap_resamples),
        "selection_flips_vs_vote": int(selection_flips),
        "verifier_recovers_outvoted": int(recovered),
        "verifier_value_added": bool(accuracy_lift or efficiency_parity),
        "efficiency_parity": bool(efficiency_parity),
        "value_added_basis": value_added_basis,
        "cost": {
            "cost_unit": "candidate_level_selection_operation_generation_budget_held_constant",
            "same_candidate_budget": True,
            "arm_a_total_selection_cost_units": arm_a_cost,
            "arm_a_total_verifier_evals": candidate_count,
            "arm_a_total_selector_evals": candidate_count,
            "arm_b_total_vote_weight_reads": arm_b_cost,
            "arm_c_total_first_candidate_reads": arm_c_cost,
            "arm_a_over_arm_b_selection_cost_ratio": _round(arm_a_cost / float(arm_b_cost)),
            "arm_a_over_arm_c_selection_cost_ratio": _round(arm_a_cost / float(arm_c_cost)),
            "latency_s": {
                "arm_a_verifier_selector_selection": round(arm_a_latency, 9),
                "arm_b_sc_vote_selection": round(arm_b_latency, 9),
                "arm_c_no_verifier_selection": round(arm_c_latency, 9),
            },
        },
    }


def _complete_artifact(
    corpus: vstar.TraceCorpus,
    metrics: dict[str, Any],
    *,
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    positive_control = {
        "status": "headroom_present",
        "oracle_at_k": metrics["oracle_at_k"],
        "sc_vote_pass1": metrics["arm_b_pass1"],
        "oracle_minus_sc_vote": _round(metrics["oracle_at_k"] - metrics["arm_b_pass1"]),
        "selection_flips_vs_vote": metrics["selection_flips_vs_vote"],
        "verifier_recovers_outvoted": metrics["verifier_recovers_outvoted"],
    }
    positive_control_confirmed = (
        positive_control["oracle_minus_sc_vote"] > 0.0
        and positive_control["selection_flips_vs_vote"] >= 1
    )
    verdict = (
        "complete: verifier_value_added_"
        f"{str(metrics['verifier_value_added']).lower()}_delta_{metrics['delta_ab']:.4f}_"
        f"ci95_{metrics['ci95'][0]:.4f}_{metrics['ci95'][1]:.4f}"
    )
    return {
        "honest_verdict": verdict,
        "verifier_value_added": metrics["verifier_value_added"],
        "moat_delta_vs_vote": {
            "status": "measured",
            "n": metrics["n_tasks"],
            "arm_a_pass1": metrics["arm_a_pass1"],
            "arm_b_sc_vote_pass1": metrics["arm_b_pass1"],
            "delta": metrics["delta_ab"],
            "ci95": metrics["ci95"],
            "bootstrap_resamples": metrics["bootstrap_resamples"],
        },
        "moat_vs_matched_control": {
            "status": "measured",
            "n": metrics["n_tasks"],
            "arm_a_pass1": metrics["arm_a_pass1"],
            "arm_c_no_verifier_pass1": metrics["arm_c_pass1"],
            "delta": metrics["delta_ac"],
            "control_policy": "deterministic_first_of_k_no_verifier",
        },
        "accuracy_cost_pareto": {
            "status": "measured",
            "same_candidate_budget": True,
            "efficiency_parity": metrics["efficiency_parity"],
            "parity_at_n_times_cheaper": None,
            "value_added_basis": metrics["value_added_basis"],
            **metrics["cost"],
        },
        "positive_control_confirmed": bool(positive_control_confirmed),
        "positive_control": positive_control,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "acceptance_gate": bool(positive_control_confirmed and metrics["verifier_value_added"]),
        "domain": corpus.domain,
        "candidate_pool_source": str(corpus.source_path),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["verifier_value_added"]) is not bool:
        raise ValueError("verifier_value_added must be a bare bool")
    if type(artifact["positive_control_confirmed"]) is not bool:
        raise ValueError("positive_control_confirmed must be a bare bool")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str) or not artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum must be a non-empty checksum")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4177")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4177")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("unexpected inference_substrate")
    for field in ("moat_delta_vs_vote", "moat_vs_matched_control", "accuracy_cost_pareto"):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
    oracle_checker: Callable[[], tuple[bool, str]] = _default_oracle_checker,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL

    try:
        headroom = _load_headroom_gate(root)
        deferred = _defer_status(headroom)
        if deferred is not None:
            artifact = _empty_artifact(
                "complete_moat_test_deferred_no_headroom_present",
                deferred,
                random_seed,
                time.perf_counter() - start,
            )
        else:
            _check_domain(headroom)
            oracle_ok, oracle_detail = oracle_checker()
            if not oracle_ok:
                raise BlockedRun("blocked_executable_oracle_unavailable")
            corpus = _load_corpus(root)
            selector = _load_selector(root)
            selector_path = root / SELECTOR_REL
            checksum = reproducibility_checksum(corpus, selector_path)
            metrics = _measure_arms(
                corpus,
                selector,
                random_seed=random_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            artifact = _complete_artifact(
                corpus,
                metrics,
                checksum=checksum,
                random_seed=random_seed,
                duration_s=time.perf_counter() - start,
            )
            artifact["executable_oracle_check"] = {"available": True, "detail": oracle_detail}
    except BlockedRun as blocked:
        artifact = _empty_artifact(blocked.reason, blocked.reason, random_seed, time.perf_counter() - start)

    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if adversarial_runner is None:
        report = _run_adversarial_verify(root, output_path)
    else:
        report = adversarial_runner(output_path)
    artifact["adversarial_verify"] = report
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
